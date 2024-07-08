"""
functions for feature map
"""
import torch
## orbit sample
def orbit_sample(orbit, samples):
    """Pick the sample belonging to the given orbit"""
    orbit_sample_dict = {}
    set_orbit =  sorted(orbit)
    for k in list(samples.keys()):
        temp = list(filter(lambda x:x!=0, list(k)))
        temp = sorted(temp)
        if temp == set_orbit:
            orbit_sample_dict[k] = samples[k]
    return orbit_sample_dict
## event sample
def event_sample(k, n, samples):
    """
    Pick the sample belonging to the given even E_{k,n}
    k: total number of photons in all modes
    n: maximum number of photons for single mode+1
    """

    orbit_list = integer_partition(k, n-1)
    orbit_list_sort = [sorted(orbit) for orbit in orbit_list]
    orbit_sample_list = [{} for i in range(len(orbit_list))]
    for i in list(samples.keys()):
        temp = list(filter(lambda x:x!=0, list(i)))
        temp = sorted(temp)
        if temp in orbit_list_sort:
            idx = orbit_list_sort.index(temp)
            orbit_sample_list[idx][i] = samples[i]
    return orbit_sample_list

def integer_partition(m,n):
    """ integer partition"""
    results = [ ]
    def back_trace(m, n, result=[]):
        if m == 0:  # 如果 m 等于 0，说明已经找到了一个分解方式，将其加入结果列表
            results.append(result)
            return
        if m < 0 or n == 0:  # 如果 m 小于 0 或者 n 等于 0，说明无法找到满足条件的分解方式，直接返回
            return
        back_trace(m, n - 1, result)  # 不使用 n 进行分解，继续递归
        back_trace(m - n, n, result + [n])  # 使用 n 进行分解，继续递归
    back_trace(m, n)
    return results

## feature map
def feature_map_event_sample(event_photon_numbers, n, samples):
    """Map a set of graph G to the feature vectors using the event examples."""
    all_feature_vectors = [ ]
    for sample in samples:
        count_list = [ ]
        total_num_samples = sum(sample.values())
        for k in event_photon_numbers:
            e_k_n = event_sample(k, n, sample)
            temp_sum = 0
            for i in range(len(e_k_n)):
                temp_sum = temp_sum+sum(e_k_n[i].values())
            count_list.append(temp_sum)
        feature_vector  =  (torch.stack(count_list)/total_num_samples).reshape(1,-1)
        all_feature_vectors.append(feature_vector.squeeze())
    return torch.stack(all_feature_vectors)
