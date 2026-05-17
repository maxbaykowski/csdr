add_executable(csdr-regression-tests regression.cpp)
target_link_libraries(csdr-regression-tests csdr)
add_test(NAME csdr-regression-tests COMMAND csdr-regression-tests)
