#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <custatevec.h>
#include <pybind11/complex.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Adjoint kernels defined in adjoint_kernels.cu.
extern "C" void dq_pauli_action(const void*, void*, uint64_t, uint64_t, uint64_t, int, const float*, cudaStream_t);
extern "C" void dq_pauli_contract(const void*, const void*, float*, uint64_t, uint64_t, uint64_t, int, uint64_t,
                                  cudaStream_t);

namespace py = pybind11;
namespace {
void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": CUDA status " + std::to_string(static_cast<int>(status)) +
                             " (" + cudaGetErrorString(status) + ")");
}
void check_custatevec(custatevecStatus_t status, const char* operation) {
  if (status != CUSTATEVEC_STATUS_SUCCESS)
    // ixquantum 1.8.0 does not implement custatevecGetErrorString reliably.
    throw std::runtime_error(std::string(operation) + ": ixquantum status " + std::to_string(static_cast<int>(status)));
}

// Own native resources independently of executor construction and tensor allocation.
struct NativeResources {
  explicit NativeResources(int device) : device(device) {}
  NativeResources(const NativeResources&) = delete;
  NativeResources& operator=(const NativeResources&) = delete;
  ~NativeResources() {
    try {
      const c10::cuda::CUDAGuard guard(device);
      close();
    } catch (...) {
      // Destructors cannot report runtime shutdown errors.
    }
  }

  void initialize() {
    const c10::cuda::CUDAGuard guard(device);
    check_custatevec(custatevecCreate(&handle), "custatevecCreate");
    check_cuda(cudaEventCreateWithFlags(&torch_to_vendor, cudaEventDisableTiming), "create torch-to-vendor event");
    check_cuda(cudaEventCreateWithFlags(&vendor_to_torch, cudaEventDisableTiming), "create vendor-to-torch event");
  }

  void close() {
    // Attempt every release before reporting an error; subsequent calls are harmless.
    const auto sampler_status =
        sampler ? custatevecSamplerDestroy(std::exchange(sampler, nullptr)) : CUSTATEVEC_STATUS_SUCCESS;
    const auto handle_status = handle ? custatevecDestroy(std::exchange(handle, nullptr)) : CUSTATEVEC_STATUS_SUCCESS;
    const auto input_status = torch_to_vendor ? cudaEventDestroy(std::exchange(torch_to_vendor, nullptr)) : cudaSuccess;
    const auto output_status =
        vendor_to_torch ? cudaEventDestroy(std::exchange(vendor_to_torch, nullptr)) : cudaSuccess;
    check_custatevec(sampler_status, "custatevecSamplerDestroy");
    check_custatevec(handle_status, "custatevecDestroy");
    check_cuda(input_status, "destroy torch-to-vendor event");
    check_cuda(output_status, "destroy vendor-to-torch event");
  }

  int device;
  custatevecHandle_t handle{nullptr};
  custatevecSamplerDescriptor_t sampler{nullptr};
  cudaEvent_t torch_to_vendor{nullptr}, vendor_to_torch{nullptr};
};

// Values match the plan constants in compiler.py.
enum class OpCode { Matrix = 0, PauliRotation = 1, Phase = 2, Permutation = 3 };
enum class GradientKind { None = 0, PauliGenerator = 1, Phase = 2 };

struct Op {
  OpCode opcode{};
  std::vector<int32_t> targets, controls, values;
  torch::Tensor matrix, inverse_matrix;
  bool diagonal_only = false, permutation_only = false;
  std::vector<custatevecIndex_t> permutations[2];
  std::vector<c10::complex<float>> diagonals[2];
  double angle{};
  std::vector<int> paulis;
  int parameter_index{-1};
  double parameter_scale{1.0};
  GradientKind gradient_kind{};
  double generator_coeff{};
};

class GPUCircuitExecutor {
  // Lease scratch from the PyTorch allocator for the current stream.
  struct WorkspaceLease {
    c10::cuda::CUDAGuard guard;
    GPUCircuitExecutor& owner;
    bool acquired = false;
    explicit WorkspaceLease(GPUCircuitExecutor& value) : guard(value.device_), owner(value) {
      owner.bind_current_stream();
      if (owner.workspace_bytes_ && !owner.workspace_.defined()) {
        acquired = true;
        owner.workspace_ =
            torch::empty({static_cast<int64_t>(owner.workspace_bytes_)},
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, owner.device_));
      }
    }
    ~WorkspaceLease() {
      if (acquired) owner.workspace_ = torch::Tensor();
    }
  };

 public:
  GPUCircuitExecutor(int nqubit, int device) : nqubit_(nqubit), device_(device), resources_(device) {
    if (nqubit < 1 || nqubit > 30) throw std::invalid_argument("nqubit must be in [1, 30]");
    resources_.initialize();
  }
  ~GPUCircuitExecutor() {
    try {
      close();
    } catch (...) {
    }
  }

  void compile(const std::vector<int>& codes, const std::vector<std::vector<int>>& targets,
               const std::vector<std::vector<int>>& controls, const py::list& matrices,
               const std::vector<std::vector<int64_t>>& permutations,
               const std::vector<std::vector<std::complex<float>>>& diagonals, const std::vector<double>& angles,
               const std::vector<std::vector<int>>& paulis, const std::vector<int>& parameter_indices,
               const std::vector<double>& parameter_scales, const std::vector<int>& gradient_kinds,
               const std::vector<double>& generator_coeffs, const std::vector<std::vector<int>>& obs_p,
               const std::vector<std::vector<int>>& obs_b, const py::object& observable_matrix,
               const std::vector<int>& observable_matrix_bits, const py::object& initial_state) {
    if (closed_) throw std::runtime_error("executor is closed");
    const c10::cuda::CUDAGuard guard(device_);
    ++generation_;
    ++plan_version_;
    state_ready_ = false;
    compiled_ = false;
    const auto n = codes.size();
    if (permutations.size() != n || diagonals.size() != n || targets.size() != n || controls.size() != n ||
        matrices.size() != n || angles.size() != n || paulis.size() != n || parameter_indices.size() != n ||
        parameter_scales.size() != n || gradient_kinds.size() != n || generator_coeffs.size() != n)
      throw std::invalid_argument("inconsistent plan lengths");
    ops_.clear();
    workspace_bytes_ = 0;
    parameter_count_ = 0;

    prepared_generation_ = -1;
    last_parameter_values_.clear();
    initial_state_ = torch::Tensor();
    if (!initial_state.is_none()) {
      initial_state_ = py::cast<torch::Tensor>(initial_state);
      if (!initial_state_.is_cuda() || initial_state_.scalar_type() != torch::kComplexFloat ||
          !initial_state_.is_contiguous() || initial_state_.numel() != (int64_t{1} << nqubit_))
        throw std::invalid_argument("initial_state must be contiguous CUDA complex64 with 2**n amplitudes");
      initial_state_ = initial_state_.clone();
    }

    for (size_t i = 0; i < n; ++i) {
      Op op;
      op.opcode = static_cast<OpCode>(codes[i]);
      op.angle = angles[i];
      op.paulis = paulis[i];
      op.parameter_index = parameter_indices[i];
      op.parameter_scale = parameter_scales[i];
      op.gradient_kind = static_cast<GradientKind>(gradient_kinds[i]);
      op.generator_coeff = generator_coeffs[i];
      parameter_count_ = std::max(parameter_count_, op.parameter_index + 1);
      for (int value : targets[i]) op.targets.push_back(value);
      for (int value : controls[i]) {
        op.controls.push_back(value);
        op.values.push_back(1);
      }
      if (op.targets.empty()) throw std::invalid_argument("operation has no targets");
      size_t size = 0;
      if (op.opcode == OpCode::Matrix) {
        if (matrices[i].is_none()) throw std::invalid_argument("MATRIX op has no matrix");
        op.matrix = py::cast<torch::Tensor>(matrices[i]);
        if (!op.matrix.is_cuda() || op.matrix.scalar_type() != torch::kComplexFloat || !op.matrix.is_contiguous())
          throw std::invalid_argument("matrices must be contiguous CUDA complex64");
        const int64_t dim = int64_t{1} << op.targets.size();
        if (op.matrix.dim() != 2 || op.matrix.size(0) != dim || op.matrix.size(1) != dim)
          throw std::invalid_argument("matrix shape does not match target count");
        // Current ixquantum does not reliably round-trip non-symmetric ROW matrices
        // with adjoint=1. Cache U^dagger explicitly and always dispatch adjoint=0.
        op.inverse_matrix = op.matrix.conj().transpose(0, 1).contiguous();

        check_custatevec(
            custatevecApplyMatrixGetWorkspaceSize(resources_.handle, CUDA_C_32F, nqubit_, op.matrix.data_ptr(),
                                                  CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, op.targets.size(),
                                                  op.controls.size(), CUSTATEVEC_COMPUTE_32F, &size),
            "matrix workspace");
      } else if (op.opcode == OpCode::Permutation) {
        const auto dim = size_t{1} << op.targets.size();
        if (permutations[i].size() != dim || diagonals[i].size() != dim)
          throw std::invalid_argument("invalid permutation descriptor size");
        op.permutations[0].assign(permutations[i].begin(), permutations[i].end());
        op.permutations[1].resize(dim);
        op.diagonals[0].resize(dim);
        op.diagonals[1].resize(dim);
        std::vector<bool> seen(dim, false);
        for (size_t row = 0; row < dim; ++row) {
          auto col = op.permutations[0][row];
          if (col < 0 || size_t(col) >= dim || seen[col]) throw std::invalid_argument("invalid permutation");
          seen[col] = true;
          op.diagonals[0][row] = c10::complex<float>(diagonals[i][row].real(), diagonals[i][row].imag());
          op.permutations[1][col] = row;
          op.diagonals[1][col] = c10::complex<float>(diagonals[i][row].real(), -diagonals[i][row].imag());
        }
        op.diagonal_only = true;
        op.permutation_only = true;
        for (size_t row = 0; row < dim; ++row) {
          op.diagonal_only &= op.permutations[0][row] == static_cast<int64_t>(row);
          op.permutation_only &= op.diagonals[0][row] == c10::complex<float>(1, 0);
        }
        if (op.diagonal_only && op.permutation_only) continue;
        for (int inverse = 0; inverse < 2; ++inverse) {
          size_t bytes = 0;
          check_custatevec(
              custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(
                  resources_.handle, CUDA_C_32F, nqubit_, op.permutations[inverse].data(), op.diagonals[inverse].data(),
                  CUDA_C_32F, op.targets.data(), op.targets.size(), op.controls.size(), &bytes),
              "permutation workspace");
          size = std::max(size, bytes);
        }
      } else if (op.opcode == OpCode::PauliRotation) {
        if (op.paulis.size() != op.targets.size())
          throw std::invalid_argument("PauliRotation requires one Pauli per target");
        for (int pauli : op.paulis)
          if (pauli < 1 || pauli > 3) throw std::invalid_argument("invalid Pauli generator");
        if (op.parameter_index >= 0 && op.gradient_kind != GradientKind::PauliGenerator)
          throw std::invalid_argument("trainable Pauli rotations require generator gradients");
      } else if (op.opcode == OpCode::Phase) {
        if (op.targets.size() != 1) throw std::invalid_argument("phase requires one target");
        c10::complex<float> diagonal[2] = {{1, 0}, {1, 0}};
        check_custatevec(custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(
                             resources_.handle, CUDA_C_32F, nqubit_, nullptr, diagonal, CUDA_C_32F, op.targets.data(),
                             1, op.controls.size(), &size),
                         "phase workspace");
      } else
        throw std::invalid_argument("unknown opcode");

      workspace_bytes_ = std::max(workspace_bytes_, size);
      ops_.push_back(std::move(op));
    }

    obs_p_ = obs_p;
    obs_b_ = obs_b;
    if (obs_p_.size() != obs_b_.size()) throw std::invalid_argument("inconsistent Pauli observable plan");
    observable_matrix_ = torch::Tensor();
    observable_matrix_bits_.clear();
    if (!observable_matrix.is_none()) {
      if (!obs_p_.empty()) throw std::invalid_argument("cannot mix Pauli and matrix observables");
      observable_matrix_ = py::cast<torch::Tensor>(observable_matrix);
      if (!observable_matrix_.is_cuda() || observable_matrix_.scalar_type() != torch::kComplexFloat ||
          !observable_matrix_.is_contiguous())
        throw std::invalid_argument("observable matrix must be contiguous CUDA complex64");
      if (observable_matrix_bits.empty()) throw std::invalid_argument("matrix observable has no target bits");
      const int64_t dim = int64_t{1} << observable_matrix_bits.size();
      if (observable_matrix_.dim() != 2 || observable_matrix_.size(0) != dim || observable_matrix_.size(1) != dim)
        throw std::invalid_argument("matrix observable has invalid shape");
      for (int bit : observable_matrix_bits) observable_matrix_bits_.push_back(bit);
      size_t size = 0;
      check_custatevec(custatevecComputeExpectationGetWorkspaceSize(
                           resources_.handle, CUDA_C_32F, nqubit_, observable_matrix_.data_ptr(), CUDA_C_32F,
                           CUSTATEVEC_MATRIX_LAYOUT_ROW, observable_matrix_bits_.size(), CUSTATEVEC_COMPUTE_32F, &size),
                       "ComputeExpectation workspace query");
      workspace_bytes_ = std::max(workspace_bytes_, size);
    }
    workspace_ = torch::Tensor();

    state_ready_ = false;
    compiled_ = true;
  }

  torch::Tensor execute(const torch::Tensor& parameters) {
    WorkspaceLease lease(*this);
    run_forward(parameters);
    return state_;
  }
  torch::Tensor forward_expectation(const torch::Tensor& parameters) {
    WorkspaceLease lease(*this);
    run_forward(parameters);
    if (obs_p_.empty() || observable_matrix_.defined())
      throw std::runtime_error("adjoint requires one or more Pauli observables");
    return expectation();
  }

  torch::Tensor expectation() {
    WorkspaceLease lease(*this);
    ready();
    if (!state_ready_) throw std::runtime_error("expectation requires a completed forward pass");
    bind_current_stream();
    if (observable_matrix_.defined()) {
      cuDoubleComplex output = make_cuDoubleComplex(0.0, 0.0);
      double residual = 0.0;
      torch_to_vendor();
      check_custatevec(custatevecComputeExpectation(
                           resources_.handle, state_.data_ptr(), CUDA_C_32F, nqubit_, &output, CUDA_C_64F, &residual,
                           observable_matrix_.data_ptr(), CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                           observable_matrix_bits_.data(), observable_matrix_bits_.size(), CUSTATEVEC_COMPUTE_32F,
                           workspace_bytes_ ? workspace_.data_ptr() : nullptr, workspace_bytes_),
                       "custatevecComputeExpectation");
      vendor_to_torch();
      check_cuda(cudaDeviceSynchronize(), "synchronize matrix expectation");
      if (std::abs(output.y) > 1e-5)
        throw std::runtime_error("Hermitian expectation has non-negligible imaginary part");
      return torch::full({1}, static_cast<float>(output.x),
                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_));
    }
    if (obs_p_.empty()) throw std::runtime_error("there is no compiled observable");
    return compute_pauli_expectations(state_, obs_p_, obs_b_);
  }

  torch::Tensor expectation_from_state(const torch::Tensor& state, const std::vector<std::vector<int>>& paulis,
                                       const std::vector<std::vector<int>>& bits) {
    if (closed_) throw std::runtime_error("executor is closed");
    const c10::cuda::CUDAGuard guard(device_);
    validate_external_state(state);

    return compute_pauli_expectations(state, paulis, bits);
  }

  torch::Tensor sample(int shots, const std::vector<int>& bit_ordering, const torch::Tensor& randnums) {
    const c10::cuda::CUDAGuard guard(device_);
    ready();
    if (!state_ready_) throw std::runtime_error("sample requires a completed forward pass");
    if (shots <= 0 || bit_ordering.empty() || bit_ordering.size() > static_cast<size_t>(nqubit_))
      throw std::invalid_argument("invalid sampling arguments");
    if (!randnums.device().is_cpu() || randnums.scalar_type() != torch::kFloat64 || !randnums.is_contiguous() ||
        randnums.numel() != shots)
      throw std::invalid_argument("randnums must be contiguous CPU float64 with one value per shot");
    std::vector<int32_t> bits;
    for (int bit : bit_ordering) {
      if (bit < 0 || bit >= nqubit_) throw std::invalid_argument("sampling bit is out of range");
      bits.push_back(bit);
    }
    bind_current_stream();
    torch_to_vendor();
    ensure_sampler(static_cast<uint32_t>(shots));
    check_custatevec(custatevecSamplerPreprocess(resources_.handle, resources_.sampler,
                                                 sampler_workspace_.defined() ? sampler_workspace_.data_ptr() : nullptr,
                                                 sampler_workspace_bytes_),
                     "custatevecSamplerPreprocess");
    // ixquantum 1.8.0 sampling reads preprocessing results on the host.
    // Non-default streams need an explicit completion before that API call.
    check_cuda(cudaStreamSynchronize(vendor_uses_current_stream_ ? current_stream_ : nullptr),
               "synchronize sampler preprocessing");
    auto output = torch::empty({shots}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    check_custatevec(custatevecSamplerSample(resources_.handle, resources_.sampler,
                                             reinterpret_cast<custatevecIndex_t*>(output.data_ptr<int64_t>()),
                                             bits.data(), bits.size(), randnums.data_ptr<double>(), shots,
                                             CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER),
                     "custatevecSamplerSample");
    vendor_to_torch();
    check_cuda(cudaDeviceSynchronize(), "synchronize sampler");
    return output;
  }

  torch::Tensor adjoint_vjp(const torch::Tensor& parameters, const torch::Tensor& grad_output) {
    WorkspaceLease lease(*this);
    validate_weights(grad_output);

    run_forward(parameters);
    return reverse_prepared(grad_output, generation_);
  }

  torch::Tensor adjoint_vjp_prepared(const torch::Tensor& parameters, const torch::Tensor& grad_output,
                                     int64_t expected_generation) {
    WorkspaceLease lease(*this);
    validate_parameters(parameters);
    validate_weights(grad_output);

    return reverse_prepared(grad_output, expected_generation);
  }

  py::tuple value_and_grad(const torch::Tensor& parameters, const torch::Tensor& weights, double constant) {
    WorkspaceLease lease(*this);
    validate_weights(weights);

    run_forward(parameters);
    build_weighted_adjoint_state(weights);
    auto energy = torch::real(torch::vdot(state_, adjoint_state_)) + constant;
    auto gradient = reverse_from_current_adjoint(generation_);
    return py::make_tuple(energy, gradient);
  }

  py::dict memory_report() const {
    py::dict out;
    out["live_workspace_bytes"] = workspace_.defined() ? workspace_.nbytes() : 0;
    out["state_bytes"] = state_.defined() ? state_.nbytes() : 0;
    out["adjoint_bytes"] = adjoint_state_.defined() ? adjoint_state_.nbytes() : 0;
    out["initial_snapshot_bytes"] = initial_state_.defined() ? initial_state_.nbytes() : 0;
    out["workspace_bytes"] = workspace_bytes_;
    out["sampler_workspace_bytes"] = sampler_workspace_bytes_;
    return out;
  }

  int64_t generation() const { return generation_; }
  int64_t plan_version() const { return plan_version_; }

  void close() {
    if (closed_) return;
    ++generation_;
    const c10::cuda::CUDAGuard guard(device_);
    const auto sync_status = cudaDeviceSynchronize();
    std::exception_ptr release_error;
    try {
      resources_.close();
    } catch (...) {
      release_error = std::current_exception();
    }
    sampler_max_shots_ = 0;
    sampler_workspace_bytes_ = 0;
    workspace_ = torch::Tensor();
    state_ = torch::Tensor();
    initial_state_ = torch::Tensor();
    adjoint_state_ = torch::Tensor();
    sampler_workspace_ = torch::Tensor();
    observable_matrix_ = torch::Tensor();
    observable_matrix_bits_.clear();
    ops_.clear();
    state_ready_ = false;
    closed_ = true;
    if (release_error) std::rethrow_exception(release_error);
    check_cuda(sync_status, "synchronize executor close");
  }

 private:
  void ready() const {
    if (closed_) throw std::runtime_error("executor is closed");
    if (!compiled_) throw std::runtime_error("executor has not been compiled");
  }
  void bind_current_stream() {
    current_stream_ = c10::cuda::getCurrentCUDAStream(device_).stream();
    if (!set_stream_known_unsupported_) {
      const auto status = custatevecSetStream(resources_.handle, current_stream_);
      if (status == CUSTATEVEC_STATUS_SUCCESS) {
        vendor_uses_current_stream_ = true;
        return;
      }
      if (status != CUSTATEVEC_STATUS_NOT_SUPPORTED) check_custatevec(status, "set stream");
      set_stream_known_unsupported_ = true;
    }
    vendor_uses_current_stream_ = false;
  }
  void torch_to_vendor() {
    if (vendor_uses_current_stream_ || current_stream_ == nullptr) return;
    check_cuda(cudaEventRecord(resources_.torch_to_vendor, current_stream_), "record torch-to-vendor event");
    check_cuda(cudaStreamWaitEvent(nullptr, resources_.torch_to_vendor, 0), "wait torch-to-vendor event");
  }
  void vendor_to_torch() {
    if (vendor_uses_current_stream_ || current_stream_ == nullptr) return;
    check_cuda(cudaEventRecord(resources_.vendor_to_torch, nullptr), "record vendor-to-torch event");
    check_cuda(cudaStreamWaitEvent(current_stream_, resources_.vendor_to_torch, 0), "wait vendor-to-torch event");
  }
  void validate_parameters(const torch::Tensor& parameters) const {
    if (!parameters.is_cuda() || parameters.get_device() != device_ || parameters.scalar_type() != torch::kFloat ||
        !parameters.is_contiguous() || parameters.dim() != 1 || parameters.numel() != parameter_count_)
      throw std::invalid_argument(
          "parameters must be a contiguous CUDA float32 vector with logical_parameter_count elements");
  }
  void validate_weights(const torch::Tensor& weights) const {
    ready();
    if (!weights.is_cuda() || weights.get_device() != device_ || weights.scalar_type() != torch::kFloat ||
        !weights.is_contiguous() || weights.dim() != 1 || weights.numel() != static_cast<int64_t>(obs_p_.size()))
      throw std::invalid_argument(
          "coefficients must be a contiguous CUDA float32 vector matching the Pauli observable count");
  }
  void validate_external_state(const torch::Tensor& state) const {
    if (!state.is_cuda() || state.get_device() != device_ || state.scalar_type() != torch::kComplexFloat ||
        state.dim() != 1 || !state.is_contiguous() || state.requires_grad() || state.numel() != (int64_t{1} << nqubit_))
      throw std::invalid_argument(
          "state must be a contiguous non-differentiable CUDA complex64 vector on the executor device with 2**n "
          "amplitudes");
  }
  torch::Tensor compute_pauli_expectations(const torch::Tensor& state,
                                           const std::vector<std::vector<int>>& observable_paulis,
                                           const std::vector<std::vector<int>>& observable_bits) {
    if (observable_paulis.empty() || observable_paulis.size() != observable_bits.size())
      throw std::invalid_argument("Pauli observables must be non-empty and have consistent plan lengths");
    std::vector<std::vector<custatevecPauli_t>> paulis(observable_paulis.size());
    std::vector<std::vector<int32_t>> bits(observable_bits.size());
    std::vector<const custatevecPauli_t*> pauli_ptrs;
    std::vector<const int32_t*> bit_ptrs;
    std::vector<uint32_t> lengths;
    std::vector<double> output(observable_paulis.size());
    for (size_t i = 0; i < observable_paulis.size(); ++i) {
      if (observable_paulis[i].empty() || observable_paulis[i].size() != observable_bits[i].size())
        throw std::invalid_argument("each Pauli observable must have matching non-empty Pauli and bit lists");
      for (int value : observable_paulis[i]) {
        if (value < 1 || value > 3) throw std::invalid_argument("invalid Pauli observable value");
        paulis[i].push_back(static_cast<custatevecPauli_t>(value));
      }
      for (int value : observable_bits[i]) {
        if (value < 0 || value >= nqubit_) throw std::invalid_argument("Pauli observable bit is out of range");
        bits[i].push_back(value);
      }
      pauli_ptrs.push_back(paulis[i].data());
      bit_ptrs.push_back(bits[i].data());
      lengths.push_back(bits[i].size());
    }
    bind_current_stream();
    torch_to_vendor();
    check_custatevec(custatevecComputeExpectationsOnPauliBasis(resources_.handle, state.data_ptr(), CUDA_C_32F, nqubit_,
                                                               output.data(), pauli_ptrs.data(), output.size(),
                                                               bit_ptrs.data(), lengths.data()),
                     "custatevecComputeExpectationsOnPauliBasis");
    vendor_to_torch();
    check_cuda(cudaDeviceSynchronize(), "synchronize Pauli expectation");
    return torch::from_blob(output.data(), {static_cast<int64_t>(output.size())},
                            torch::TensorOptions().dtype(torch::kFloat64))
        .clone()
        .to(torch::kFloat32)
        .to(torch::Device(torch::kCUDA, device_));
  }
  std::vector<float> parameter_values(const torch::Tensor& parameters) {
    validate_parameters(parameters);

    auto host = parameters.to(torch::kCPU).contiguous();
    const float* ptr = host.data_ptr<float>();
    return std::vector<float>(ptr, ptr + host.numel());
  }
  void prepare_initial() {
    if (!state_.defined())
      state_ = torch::empty({int64_t{1} << nqubit_},
                            torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA, device_));
    if (initial_state_.defined()) {
      state_.copy_(initial_state_);
      return;
    }
    state_.zero_();
    state_.index({0}).fill_(c10::complex<float>(1, 0));
  }
  void run_forward(const torch::Tensor& parameters) {
    ready();
    bind_current_stream();
    ++generation_;
    state_ready_ = false;
    last_parameter_values_ = parameter_values(parameters);
    prepare_initial();
    for (const auto& op : ops_) {
      if (op.opcode == OpCode::Matrix || op.opcode == OpCode::Permutation)
        apply_matrix(state_, op, false);
      else if (op.opcode == OpCode::PauliRotation)
        apply_rotation(state_, op,
                       op.parameter_index >= 0 ? -0.5 * op.parameter_scale * last_parameter_values_[op.parameter_index]
                                               : op.angle);
      else if (op.opcode == OpCode::Phase)
        apply_phase(
            state_, op,
            op.parameter_index >= 0 ? op.parameter_scale * last_parameter_values_[op.parameter_index] : op.angle);
    }
    state_ready_ = true;
    prepared_generation_ = generation_;
  }
  void ensure_sampler(uint32_t shots) {
    if (resources_.sampler != nullptr) {
      if (sampler_max_shots_ >= shots) return;
      sampler_max_shots_ = 0;
      check_custatevec(custatevecSamplerDestroy(std::exchange(resources_.sampler, nullptr)),
                       "custatevecSamplerDestroy");
    }
    size_t size = 0;
    check_custatevec(custatevecSamplerCreate(resources_.handle, state_.data_ptr(), CUDA_C_32F, nqubit_,
                                             &resources_.sampler, shots, &size),
                     "custatevecSamplerCreate");
    sampler_workspace_ = size ? torch::empty({static_cast<int64_t>(size)},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_))
                              : torch::Tensor();
    sampler_workspace_bytes_ = size;
    sampler_max_shots_ = shots;
  }
  void apply_matrix(torch::Tensor& state, const Op& op, bool adjoint) {
    if (op.opcode == OpCode::Permutation) {
      torch_to_vendor();
      auto& permutation = op.permutations[adjoint ? 1 : 0];
      check_custatevec(custatevecApplyGeneralizedPermutationMatrix(
                           resources_.handle, state.data_ptr(), CUDA_C_32F, nqubit_,
                           op.diagonal_only ? nullptr : const_cast<custatevecIndex_t*>(permutation.data()),
                           op.permutation_only ? nullptr : op.diagonals[adjoint ? 1 : 0].data(), CUDA_C_32F, 0,
                           op.targets.data(), op.targets.size(), op.controls.empty() ? nullptr : op.controls.data(),
                           op.values.empty() ? nullptr : op.values.data(), op.controls.size(),
                           workspace_bytes_ ? workspace_.data_ptr() : nullptr, workspace_bytes_),
                       "structured matrix");
      vendor_to_torch();
      return;
    }

    torch_to_vendor();

    const auto& matrix = adjoint ? op.inverse_matrix : op.matrix;
    check_custatevec(custatevecApplyMatrix(resources_.handle, state.data_ptr(), CUDA_C_32F, nqubit_, matrix.data_ptr(),
                                           CUDA_C_32F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, op.targets.data(),
                                           op.targets.size(), op.controls.empty() ? nullptr : op.controls.data(),
                                           op.values.empty() ? nullptr : op.values.data(), op.controls.size(),
                                           CUSTATEVEC_COMPUTE_32F, workspace_bytes_ ? workspace_.data_ptr() : nullptr,
                                           workspace_bytes_),
                     "custatevecApplyMatrix");
    vendor_to_torch();
  }
  void apply_phase(torch::Tensor& state, const Op& op, double angle) {
    c10::complex<float> diagonal[2] = {{1, 0}, {float(std::cos(angle)), float(std::sin(angle))}};

    torch_to_vendor();
    check_custatevec(custatevecApplyGeneralizedPermutationMatrix(
                         resources_.handle, state.data_ptr(), CUDA_C_32F, nqubit_, nullptr, diagonal, CUDA_C_32F, 0,
                         op.targets.data(), 1, op.controls.empty() ? nullptr : op.controls.data(),
                         op.values.empty() ? nullptr : op.values.data(), op.controls.size(),
                         workspace_bytes_ ? workspace_.data_ptr() : nullptr, workspace_bytes_),
                     "phase diagonal");
    vendor_to_torch();
  }
  void apply_rotation(torch::Tensor& state, const Op& op, double angle) {
    torch_to_vendor();
    std::vector<custatevecPauli_t> paulis;
    for (int pauli : op.paulis) paulis.push_back(static_cast<custatevecPauli_t>(pauli));
    check_custatevec(custatevecApplyPauliRotation(resources_.handle, state.data_ptr(), CUDA_C_32F, nqubit_, angle,
                                                  paulis.data(), op.targets.data(), op.targets.size(),
                                                  op.controls.empty() ? nullptr : op.controls.data(),
                                                  op.values.empty() ? nullptr : op.values.data(), op.controls.size()),
                     "custatevecApplyPauliRotation");
    vendor_to_torch();
  }
  void apply_inverse(torch::Tensor& state, const Op& op, double theta) {
    if (op.opcode == OpCode::Matrix || op.opcode == OpCode::Permutation)
      apply_matrix(state, op, true);
    else if (op.opcode == OpCode::PauliRotation)
      apply_rotation(state, op, op.parameter_index >= 0 ? 0.5 * op.parameter_scale * theta : -op.angle);
    else if (op.opcode == OpCode::Phase)
      apply_phase(state, op, -(op.parameter_index >= 0 ? op.parameter_scale * theta : op.angle));
  }
  void build_weighted_adjoint_state(const torch::Tensor& weights) {
    bind_current_stream();
    if (!adjoint_state_.defined()) adjoint_state_ = torch::empty_like(state_);
    adjoint_state_.zero_();
    for (int64_t index = 0; index < weights.numel(); ++index) {
      uint64_t x = 0, z = 0;
      int num_y = 0;
      for (size_t j = 0; j < obs_p_[index].size(); ++j) {
        int pauli = obs_p_[index][j], bit = obs_b_[index][j];
        if (pauli == 1 || pauli == 2) x |= uint64_t{1} << bit;
        if (pauli == 2 || pauli == 3) z |= uint64_t{1} << bit;
        if (pauli == 2) ++num_y;
      }
      dq_pauli_action(state_.data_ptr(), adjoint_state_.data_ptr(), state_.numel(), x, z, num_y,
                      weights.data_ptr<float>() + index, current_stream_);
      check_cuda(cudaGetLastError(), "adjoint Pauli accumulation");
    }
  }
  torch::Tensor reverse_prepared(const torch::Tensor& weights, int64_t expected_generation) {
    if (!state_ready_ || prepared_generation_ != expected_generation || generation_ != expected_generation)
      throw std::runtime_error("prepared adjoint state is stale; execute forward again");
    build_weighted_adjoint_state(weights);
    return reverse_from_current_adjoint(expected_generation);
  }
  torch::Tensor reverse_from_current_adjoint(int64_t expected_generation) {
    if (!state_ready_ || prepared_generation_ != expected_generation || generation_ != expected_generation)
      throw std::runtime_error("prepared adjoint state is stale; execute forward again");
    bind_current_stream();
    auto gradients =
        torch::zeros({parameter_count_}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, device_));

    // A block-reduction buffer is reused in stream order for all parameter contributions.
    auto partial = torch::empty({1024}, gradients.options());
    for (auto iterator = ops_.rbegin(); iterator != ops_.rend(); ++iterator) {
      const Op& op = *iterator;
      const double theta = op.parameter_index >= 0 ? last_parameter_values_[op.parameter_index] : 0.0;
      uint64_t controls = 0;
      for (int bit : op.controls) controls |= uint64_t{1} << bit;
      if (op.parameter_index >= 0 && op.gradient_kind == GradientKind::Phase) {
        // Phase generator: project onto target=1 and the active control sector.
        uint64_t projector = controls | (uint64_t{1} << op.targets[0]);
        dq_pauli_contract(state_.data_ptr(), adjoint_state_.data_ptr(), partial.data_ptr<float>(), state_.numel(), 0, 0,
                          0, projector, current_stream_);
        check_cuda(cudaGetLastError(), "phase gradient contraction");
        gradients.index({op.parameter_index}).add_(partial.sum() * (-2.0 * op.parameter_scale));
      }
      if (op.parameter_index >= 0 && op.gradient_kind == GradientKind::PauliGenerator) {
        uint64_t x = 0, z = 0;
        int num_y = 0;
        for (size_t j = 0; j < op.paulis.size(); ++j) {
          int pauli = op.paulis[j], bit = op.targets[j];
          if (pauli == 1 || pauli == 2) x |= uint64_t{1} << bit;
          if (pauli == 2 || pauli == 3) z |= uint64_t{1} << bit;
          if (pauli == 2) ++num_y;
        }
        dq_pauli_contract(state_.data_ptr(), adjoint_state_.data_ptr(), partial.data_ptr<float>(), state_.numel(), x, z,
                          num_y, controls, current_stream_);
        check_cuda(cudaGetLastError(), "Pauli gradient contraction");
        gradients.index({op.parameter_index}).add_(partial.sum() * (-2.0 * op.generator_coeff));
      }
      apply_inverse(state_, op, theta);
      apply_inverse(adjoint_state_, op, theta);
    }
    state_ready_ = false;
    prepared_generation_ = -1;
    ++generation_;
    return gradients;
  }

  int nqubit_, device_;
  NativeResources resources_;
  cudaStream_t current_stream_{nullptr};
  bool set_stream_known_unsupported_{false}, vendor_uses_current_stream_{false};
  torch::Tensor state_, initial_state_, adjoint_state_, workspace_, sampler_workspace_, observable_matrix_;
  size_t workspace_bytes_{0}, sampler_workspace_bytes_{0};
  uint32_t sampler_max_shots_{0};
  int parameter_count_{0};
  int64_t generation_{0}, plan_version_{0}, prepared_generation_{-1};
  std::vector<float> last_parameter_values_;
  std::vector<Op> ops_;
  std::vector<std::vector<int>> obs_p_, obs_b_;
  std::vector<int32_t> observable_matrix_bits_;
  bool compiled_{false}, closed_{false}, state_ready_{false};
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<GPUCircuitExecutor>(module, "GPUCircuitExecutor")
      .def(py::init<int, int>())
      .def("compile", &GPUCircuitExecutor::compile)
      .def("execute", &GPUCircuitExecutor::execute)
      .def("forward_expectation", &GPUCircuitExecutor::forward_expectation)
      .def("expectation", &GPUCircuitExecutor::expectation)
      .def("expectation_from_state", &GPUCircuitExecutor::expectation_from_state)
      .def("adjoint_vjp", &GPUCircuitExecutor::adjoint_vjp)
      .def("adjoint_vjp_prepared", &GPUCircuitExecutor::adjoint_vjp_prepared)
      .def("value_and_grad", &GPUCircuitExecutor::value_and_grad)

      .def("memory_report", &GPUCircuitExecutor::memory_report)

      .def("sample", &GPUCircuitExecutor::sample)
      .def("generation", &GPUCircuitExecutor::generation)
      .def("plan_version", &GPUCircuitExecutor::plan_version)
      .def("close", &GPUCircuitExecutor::close);
}
