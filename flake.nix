{
  description = "A C/C++ implementation of CNN";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        compilers = with pkgs; [ gcc cudaPackages.cuda_nvcc ];
        libraries = with pkgs; [
          pkg-config

          cudaPackages.cuda_cudart
          cudaPackages.libcublas
          cudaPackages.libcurand

          openblas

          nlohmann_json
        ];
        tools = with pkgs; [ clang-tools ];
      in with pkgs; {
        devShells.default =
          mkShell { buildInputs = compilers ++ libraries ++ tools; };

        packages = rec {
          default = cnn_trainer;

          cnn_trainer = let bin = "mnist_cnn_trainer";
          in stdenv.mkDerivation {
            pname = "fedcnn_cnn_trainer";
            version = "0.1.0";

            src = ./.;

            buildInputs = compilers ++ libraries;

            buildPhase = ''
              make mnist_cnn_trainer
            '';

            installPhase = ''
              install -Dm755 ${bin} $out/bin/${bin}
            '';
          };
        };
      });
}
