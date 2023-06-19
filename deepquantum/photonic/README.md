# Todo


- 参考他的参数放置顺序 https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.cu.html
- batch broadcast move inside *_matrix funcion

- 门支持三种类型的
```
d = Displacement(r=tensor(1.0), phi=tensor(0.0), mode)
d = Displacement(r=Parameter, phi=Parameter, mode)
d = Displacement(r=Parameter, phi=tensor(0.0), mode)

d.set_r
d.set_phi
```

- 训练好的模型 要支持 不同batch_size的推理

- 使用 to(device） 调用GPU

- 任意变换 batch_size

