{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane = {
      url = "github:ipetkov/crane";
    };
  };

  outputs = {
    self,
    nixpkgs,
    fenix,
    crane,
    ...
  }: let
    systems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forAllSystems = f:
      nixpkgs.lib.genAttrs systems (system:
        f {
          pkgs = nixpkgs.legacyPackages.${system};
          fenixPkgs = fenix.packages.${system};
          craneLib =
            (crane.mkLib nixpkgs.legacyPackages.${system}).overrideToolchain
            fenix.packages.${system}.stable.toolchain;
        });

    perSystem = forAllSystems ({
      pkgs,
      fenixPkgs,
      craneLib,
    }: let
      src = craneLib.cleanCargoSource ./.;
      onnxruntime-bin = pkgs.callPackage ./nix/onnxruntime.nix {};
      commonArgs = {
        inherit src;
        pname = "mcp-server-qdrant";
        strictDeps = true;
        nativeBuildInputs = pkgs.lib.optionals pkgs.stdenv.isLinux [
          pkgs.pkg-config
          pkgs.openssl
        ];
        buildInputs =
          pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.openssl
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
            pkgs.apple-sdk_15
          ];
      };
      meta = {
        description = "Rust MCP server for Qdrant with local embeddings";
        license = pkgs.lib.licenses.asl20;
        mainProgram = "mcp-server-qdrant";
      };
      onnxSupported = builtins.elem pkgs.stdenv.hostPlatform.system [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      ext = pkgs.stdenv.hostPlatform.extensions.sharedLibrary;

      # Candle
      candleArgs =
        commonArgs
        // {
          cargoExtraArgs = "--no-default-features --features candle";
        };
      candleArtifacts = craneLib.buildDepsOnly candleArgs;
      candlePkg = craneLib.buildPackage (candleArgs
        // {
          cargoArtifacts = candleArtifacts;
          doCheck = false;
          inherit meta;
        });

      # ONNX
      onnxArgs =
        commonArgs
        // {
          cargoExtraArgs = "--no-default-features --features onnx";
          ORT_DYLIB_PATH = "${onnxruntime-bin}/lib/libonnxruntime${ext}";
        };
      onnxArtifacts = craneLib.buildDepsOnly onnxArgs;
      onnxPkg = let
        unwrapped = craneLib.buildPackage (onnxArgs
          // {
            cargoArtifacts = onnxArtifacts;
            doCheck = false;
            inherit meta;
          });
      in
        pkgs.symlinkJoin {
          name = "mcp-server-qdrant";
          paths = [unwrapped];
          nativeBuildInputs = [pkgs.makeWrapper];
          postBuild = ''
            wrapProgram $out/bin/mcp-server-qdrant \
              --set ORT_DYLIB_PATH "${onnxruntime-bin}/lib/libonnxruntime${ext}"
          '';
          inherit meta;
        };

      toolchain = fenixPkgs.stable.withComponents [
        "cargo"
        "clippy"
        "rustc"
        "rustfmt"
        "rust-src"
        "rust-analyzer"
      ];

      # Extended toolchain for the dev shell — adds cross targets
      devToolchain =
        if pkgs.stdenv.isLinux
        then
          fenixPkgs.combine [
            fenixPkgs.stable.cargo
            fenixPkgs.stable.clippy
            fenixPkgs.stable.rustc
            fenixPkgs.stable.rustfmt
            fenixPkgs.stable.rust-src
            fenixPkgs.stable.rust-analyzer
            fenixPkgs.targets."x86_64-unknown-linux-musl".stable.rust-std
            fenixPkgs.targets."aarch64-unknown-linux-musl".stable.rust-std
          ]
        else if pkgs.stdenv.isDarwin && pkgs.stdenv.isAarch64
        then
          fenixPkgs.combine [
            fenixPkgs.stable.cargo
            fenixPkgs.stable.clippy
            fenixPkgs.stable.rustc
            fenixPkgs.stable.rustfmt
            fenixPkgs.stable.rust-src
            fenixPkgs.stable.rust-analyzer
            fenixPkgs.targets."x86_64-apple-darwin".stable.rust-std
          ]
        else toolchain;
    in {
      packages =
        {candle = candlePkg;}
        // pkgs.lib.optionalAttrs onnxSupported {
          default = onnxPkg;
          onnx = onnxPkg;
          onnxruntime = onnxruntime-bin;
        }
        // pkgs.lib.optionalAttrs (!onnxSupported) {
          default = candlePkg;
        };

      checks =
        {
          fmt = craneLib.cargoFmt {
            inherit src;
            pname = "mcp-server-qdrant";
          };

          candle-clippy = craneLib.cargoClippy (candleArgs
            // {
              cargoArtifacts = candleArtifacts;
              cargoClippyExtraArgs = "--workspace -- -D warnings";
            });

          taplo =
            pkgs.runCommand "taplo-check" {
              nativeBuildInputs = [pkgs.taplo];
            } ''
              cd ${self}
              taplo check
              touch $out
            '';

          typos =
            pkgs.runCommand "typos-check" {
              nativeBuildInputs = [pkgs.typos];
            } ''
              cd ${self}
              typos
              touch $out
            '';

          nix-fmt =
            pkgs.runCommand "nix-fmt-check" {
              nativeBuildInputs = [pkgs.alejandra];
            } ''
              alejandra --check ${self}/flake.nix ${self}/nix/
              touch $out
            '';
        }
        // pkgs.lib.optionalAttrs onnxSupported {
          onnx-clippy = craneLib.cargoClippy (onnxArgs
            // {
              cargoArtifacts = onnxArtifacts;
              cargoClippyExtraArgs = "--workspace -- -D warnings";
            });
        };

      devShells.default = pkgs.mkShell {
        packages =
          [
            devToolchain
            pkgs.just
            pkgs.taplo
            pkgs.typos
            pkgs.actionlint
            pkgs.cargo-nextest
            pkgs.qdrant
            pkgs.maturin
            pkgs.python3
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            pkgs.pkg-config
            pkgs.openssl
            pkgs.pkgsCross.musl64.stdenv.cc
            pkgs.pkgsCross.aarch64-multiplatform-musl.stdenv.cc
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.apple-sdk_15
          ];

        env =
          {
            RUST_BACKTRACE = "1";
            RUST_SRC_PATH = "${devToolchain}/lib/rustlib/src/rust/library";
          }
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.openssl pkgs.stdenv.cc.cc.lib];
            CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER = "${pkgs.pkgsCross.musl64.stdenv.cc}/bin/${pkgs.pkgsCross.musl64.stdenv.cc.targetPrefix}cc";
            CC_x86_64_unknown_linux_musl = "${pkgs.pkgsCross.musl64.stdenv.cc}/bin/${pkgs.pkgsCross.musl64.stdenv.cc.targetPrefix}cc";
            CFLAGS_x86_64_unknown_linux_musl = "-U_FORTIFY_SOURCE";
            CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER = "${pkgs.pkgsCross.aarch64-multiplatform-musl.stdenv.cc}/bin/${pkgs.pkgsCross.aarch64-multiplatform-musl.stdenv.cc.targetPrefix}cc";
            CC_aarch64_unknown_linux_musl = "${pkgs.pkgsCross.aarch64-multiplatform-musl.stdenv.cc}/bin/${pkgs.pkgsCross.aarch64-multiplatform-musl.stdenv.cc.targetPrefix}cc";
            CFLAGS_aarch64_unknown_linux_musl = "-U_FORTIFY_SOURCE";
            X86_64_UNKNOWN_LINUX_MUSL_OPENSSL_STATIC = "1";
            X86_64_UNKNOWN_LINUX_MUSL_OPENSSL_LIB_DIR = "${pkgs.pkgsCross.musl64.openssl.out}/lib";
            X86_64_UNKNOWN_LINUX_MUSL_OPENSSL_INCLUDE_DIR = "${pkgs.pkgsCross.musl64.openssl.dev}/include";
            AARCH64_UNKNOWN_LINUX_MUSL_OPENSSL_STATIC = "1";
            AARCH64_UNKNOWN_LINUX_MUSL_OPENSSL_LIB_DIR = "${pkgs.pkgsCross.aarch64-multiplatform-musl.openssl.out}/lib";
            AARCH64_UNKNOWN_LINUX_MUSL_OPENSSL_INCLUDE_DIR = "${pkgs.pkgsCross.aarch64-multiplatform-musl.openssl.dev}/include";
          }
          // pkgs.lib.optionalAttrs onnxSupported {
            ORT_DYLIB_PATH = onnxArgs.ORT_DYLIB_PATH;
          };
      };
    });
  in {
    formatter = nixpkgs.lib.genAttrs systems (system: nixpkgs.legacyPackages.${system}.alejandra);

    overlays.default = final: _prev: {
      mcp-server-qdrant = self.packages.${final.stdenv.hostPlatform.system}.default;
    };

    packages = nixpkgs.lib.mapAttrs (_: s: s.packages) perSystem;
    checks = nixpkgs.lib.mapAttrs (_: s: s.checks) perSystem;
    devShells = nixpkgs.lib.mapAttrs (_: s: s.devShells) perSystem;
  };
}
