Data structure for constant time sampling from a discrete probability distribution that changes over time.

# Compilation
```
sudo apt install gcc-11 g++-11 build-essential cmake

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```

# Usage Example
```
#include <iostream>
#include <random>
#include <sampling/DynamicProposalArray.hpp>

int main() {
   std::random_device rd;
   size_t seed = rd();
   std::mt19937_64 gen(seed);

   const std::vector<double> weights = {5.0, 1.5, 0.1, 2.5, 0.9};
   DynamicProposalArray pa(weights);
   std::vector<size_t> counts(weights.size(), 0);
   for (size_t s = 0; s < 1000; ++s) counts[pa.sample(gen)]++;
   for (auto c : counts) std::cout << c << " ";
   std::cout << std::endl;

   const std::vector<double> updates = {1.0, 7.0, 0.5, 0.01, 1.49};
   std::vector<size_t> updated_counts(weights.size(), 0);
   for (size_t i = 0; i < updates.size(); ++i) pa.update(i, updates[i]);
   for (size_t s = 0; s < 1000; ++s) updated_counts[pa.sample(gen)]++;
   for (auto c : updated_counts) std::cout << c << " ";
   std::cout << std::endl;
   
   return 0;
}
```
Sample output:
```
495 160 10 255 80 
109 713 47 1 130 
```
