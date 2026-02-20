{
  description = "RL-Robocrane";

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      python = pkgs.python312; # or whichever version is working
      tikzplotlib = python.pkgs.buildPythonPackage rec {
        pname = "tikzplotlib";
        version = "0.10.1"; # adjust to latest
        src = pkgs.fetchFromGitHub {
          owner = "JasonGross";
          repo = "tikzplotlib";
          rev = "v0.10.1.post13"; # Use the appropriate tag or commit
          sha256 = "sha256-xymA5sMqvHhkD7XMAbvx20thcgfvwGdeRwd04TqA1SY=";
        };
        pyproject = true;
        # adjust build-system: use setuptools or poetry, if needed
        propagatedBuildInputs = with python.pkgs; [
          matplotlib
          numpy
          flit
          webcolors
        ];
        # possibly disable tests if they fail
        doCheck = false;
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        shellHook = ''
          export QT_QPA_PLATFORM=xcb
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
            with pkgs;
            lib.makeLibraryPath [
              # xorg.libX11
              # xorg.libXt
              # xorg.libSM
              # zlib
              # glib
              # udev
              libGL
              # glfw
              # boost
              # libusb1
              # gmp
              # cddlib
            ]
          }:${pkgs.stdenv.cc.cc.lib}/lib"
            source .venv/bin/activate
            export PYTHONPATH="$PYTHONPATH:$VIRTUAL_ENV/lib/python3.12/site-packages"
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBNVIDIA_PATH"
            export CODESTRAL_API_KEY=$(secret-tool lookup api/codestral password)
            unset QT_PLUGIN_PATH
        '';
        name = "RL-Robocrane";
        buildInputs = with pkgs; [
          python312Packages.triton-bin
          python312Packages.torch-bin
          python312Packages.pip
          python312Packages.tkinter
          python312Packages.cmake
          python312Packages.isort
          python312Packages.debugpy
          python312Packages.pinocchio
          python312Packages.matplotlib
          python312Packages.coal
          python312Packages.eigenpy
          python312Packages.isort
          python312Packages.debugpy
          python312Packages.mujoco
          python312Packages.scipy
          python312Packages.tqdm
          python312Packages.cloudpickle
          python312Packages.pandas
          python312Packages.colorama
          mujoco
          (pkgs.python312.withPackages (
            ps: with ps; [
              tikzplotlib
            ]
          ))
          cmake
          gcc
          stdenv
          udev
          libGL
          glfw
          graphviz
          gmp
          glibc.bin
          protobuf
          texliveFull
        ];
      };
    };
}
