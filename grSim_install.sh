# Ubuntu/VM
sudo apt install git build-essential cmake pkg-config qtbase5-dev libqt5opengl5-dev libgl1-mesa-dev libglu1-mesa-dev libprotobuf-dev protobuf-compiler libode-dev libboost-dev
sudo apt install tini libqt5opengl5 libode8 x11vnc xvfb
cd grSim/
mkdir build
cd build/
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make
cd ../..

# run grSim headless
grSim -H -platform offscreen
# grsim runs with division B by default

# MacOS ARM Series
