# LinAlgTools - Numerical Linear Algebra in C++

**LinAlgTools** is a header-only C++23 library, which focues on matrix decomposition algorithms with an emphasis on:
- **Educational value**: Well-documented theoretical and technical parts (see the [report](https://github.com/Avgustineiw/LinAlgTools/blob/dev/CP_Report.pdf))
- **Practical usability**: Implementation of numerically stable algorithms, utilizing modern C++ features

## Quick Start
```bash
# Clone and include headers in your project
git clone https://github.com/Avgustineiw/LinAlgTools.git
```

## Key Features

### Core Components
- **Matrix<T>**: Templated matrix class supporting:
  - Real and complex numbers (`float`, `double`, `std::complex`)
  - Cross-type (between matrix and views) basic arithmetic operations
  - Matrix-specific operations
- **Views**:
  - `SubMatrix<T>`: Mutable view without data copying
  - `ConstSubMatrix<T>`: Immutable view without data copying
- **Utility**:
  - Math helpers (sign, comparisons, etc.)
  - Type traits
  - Random matrix generation (dense/sparse)

### Implemented Algorithms
| Algorithm            | Method                                   | Notes                     |
|----------------------|------------------------------------------|---------------------------|
| QR Decomposition     | Householder reflections                  | Preferred for dense mats  |
|                      | Givens rotations                         | Better for sparse/structured |
| Hessenberg Form      | Householder reduction                    | Schur decomposition prep  |
| Schur Decomposition  | QR algorithm with shifts                 | Real matrices only        |
| Bidiagonalization    | Householder reflections                  | SVD prep                  |
| SVD                  | Bidiagonalization & alternating QR       | Naive implementation      |


## Using the library
The library is wrapped around namespace `LinAlgTools`, which contains:
 - namespace `Algorithm` for algorithms mentioned above
 - namespace `Core` for useful math and matrix functionality

```cpp
#include "src/algorithms/qr_decomposition.h"
#include "src/core/random_generator.h"
#include "src/types/matrix.h"

using namespace LinAlgTools;

int main() {
        // Create a random generator with a seed 20
        Core::RandomGenerator rg(20);

        // Create a random 4x4 dense matrix with entries from -5 to 5
        Matrix<double> A = rg.GetRandomDenseMatrix<double>(4, 4, -5, 5);

        // Perform QR decomposition
        auto [Q, R] = Algorithm::HouseholderQR(A);

        // Verify decomposition
        if (Core::IsOrthogonal(Q) &&
            Core::IsUpperTriangular(R)) {
                std::cout << "Valid QR decomposition\n";
        }

        return 0;
}
```

## Build and Test
Requires:
- C++23 compiler (Clang recommended)
- CMake ≥ 3.26
- GoogleTest (automatically fetched)

```bash
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build

# Run tests
cd build
./test_all  # Runs both validation and benchmarks
./test_validation  # Just correctness tests
./test_performance  # Just benchmark tests
```

## About the project
Developed as a course project for the 2nd year of the Bachelor's program in Data Science and Business Analytics at HSE University (2025). Author: Zinkin Zakhar, 2nd year student.

## License

MIT License (see [LICENSE](https://github.com/Avgustineiw/LinAlgTools/blob/dev/LICENSE))
