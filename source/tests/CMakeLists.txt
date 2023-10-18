add_executable(TestSampling TestSampling.cpp)
target_link_libraries(TestSampling libsampling)

add_executable(BenchmarkConstruction BenchmarkConstruction.cpp)
target_link_libraries(BenchmarkConstruction libsampling)

add_executable(BenchmarkSampling BenchmarkSampling.cpp)
target_link_libraries(BenchmarkSampling libsampling)

add_executable(BenchmarkDynamicIncreasing BenchmarkDynamicIncreasing.cpp)
target_link_libraries(BenchmarkDynamicIncreasing libsampling)

add_executable(BenchmarkDynamicConstant BenchmarkDynamicConstant.cpp)
target_link_libraries(BenchmarkDynamicConstant libsampling)

add_executable(BenchmarkDynamicSampling BenchmarkDynamicSampling.cpp)
target_link_libraries(BenchmarkDynamicSampling libsampling)

add_executable(BenchmarkInsertionRemoval BenchmarkInsertionRemoval.cpp)
target_link_libraries(BenchmarkInsertionRemoval libsampling)

add_executable(BenchmarkLogCascade BenchmarkLogCascade.cpp)
target_link_libraries(BenchmarkLogCascade libsampling)