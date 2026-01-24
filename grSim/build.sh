rm -rf bin/
rm -rf build/

mkdir build
cd build/

/Applications/CMake.app/Contents/bin/cmake -DCMAKE_INSTALL_PREFIX=/opt/homebrew/Cellar ..
make

cd ..