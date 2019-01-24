# single-translation-units tests
set(STU_TESTS pingpong
          simple_public simple_private simple_publish
          non_trivially_copyable)
foreach(t ${STU_TESTS})
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} gam)
    target_compile_options(${t} INTERFACE "-DGAM_LOG -DGAM_DBG")
endforeach(t)

# multi-translation-units tests
add_executable(mtu mtu_main.cpp mtu_ranks.cpp)
target_link_libraries(mtu gam)
target_compile_options(mtu INTERFACE "-DGAM_LOG -DGAM_DBG")

# test commands
set(GAMRUN ${PROJECT_SOURCE_DIR}/bin/gamrun-local)

add_test(NAME pingpong
         COMMAND ${GAMRUN} -v -n 2 -l localhost ${CMAKE_CURRENT_BINARY_DIR}/pingpong)
add_test(NAME simple_public
         COMMAND ${GAMRUN} -v -n 3 -l localhost ${CMAKE_CURRENT_BINARY_DIR}/simple_public)
add_test(NAME simple_private
         COMMAND ${GAMRUN} -v -n 3 -l localhost ${CMAKE_CURRENT_BINARY_DIR}/simple_private)
add_test(NAME simple_publish
         COMMAND ${GAMRUN} -v -n 3 -l localhost ${CMAKE_CURRENT_BINARY_DIR}/simple_publish)
add_test(NAME non_trivially_copyable
         COMMAND ${GAMRUN} -v -n 2 -l localhost ${CMAKE_CURRENT_BINARY_DIR}/non_trivially_copyable)
add_test(NAME mtu
         COMMAND ${GAMRUN} -v -n 3 -l localhost ${CMAKE_CURRENT_BINARY_DIR}/mtu)