grSim installation on Mac Arm Machines(M1 onwards)

**Very important note**: grSim will be compiled with CMake < 4, if you have CMake >= 4 you have to download v3.X.X matching your arch from this link[https://github.com/Kitware/CMake/releases].

--> install CMake 3.X.X
brew install pkg-config
brew install ode
brew install qt@5
**Note**: as grSim uses a very old version of protobuf/vartypes and brew doesn't have it, skip the installation so the CMake downloads and builds the required version.

git clone git@github.com:RoboCup-SSL/grSim.git
cd grSim/
mkdir build
cd build

./path/to/cmake3.X.X -DCMAKE_INSTALL_PREFIX=/opt/homebrew/Cellar ..
**Note**: in the grSim offical install instructions that is denoted as /usr/local which is the path on intel based machines, for installation with homebrew use /opt/homebrew/Cellar


After you install it cd ../bin/ and ./grSim to run it. (or if you want just cp it to /opt/homebrew/bin so its visible from the PATH)

**Issue**: you may see the following message repeatedly in the console log "Sending UDP datagram failed (maybe too large?). Size was: X byte(s)."
this issue is related to how MacOS handles packet sending on multicast mode, to resolve this issue set the 'Vision multicast address' to the broadcast address of your current network interface, for example if you're using wifi and network interface is "en0" just ifconfig -> look for en0 -> get the inet broadcast address and set it for vision multicast address.
You also have to configure the client to use this for vision worker.
