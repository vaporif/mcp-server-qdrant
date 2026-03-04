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
  in {
    formatter = nixpkgs.lib.genAttrs systems (system: nixpkgs.legacyPackages.${system}.alejandra);

    overlays.default = final: _prev: {
      mcp-server-qdrant = self.packages.${final.stdenv.hostPlatform.system}.default;
    };

    packages = forAllSystems ({
      pkgs,
      craneLib,
      ...
    }: let
      src = craneLib.cleanCargoSource ./.;
      onnxruntime-bin = pkgs.callPackage ./nix/onnxruntime.nix {};
      commonArgs = {
        inherit src;
        pname = "mcp-server-qdrant";
        strictDeps = true;
        buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
          pkgs.libiconv
          pkgs.apple-sdk_15
        ];
      };
      cargoArtifacts = craneLib.buildDepsOnly commonArgs;
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
    in
      {
        default = craneLib.buildPackage (commonArgs // {inherit cargoArtifacts meta;});
      }
      // pkgs.lib.optionalAttrs onnxSupported {
        onnx = let
          unwrapped = craneLib.buildPackage (commonArgs
            // {
              inherit cargoArtifacts meta;
              cargoExtraArgs = "--no-default-features --features onnx";
              ORT_DYLIB_PATH = "${onnxruntime-bin}/lib/libonnxruntime${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}";
            });
        in
          pkgs.symlinkJoin {
            name = "mcp-server-qdrant-onnx";
            paths = [unwrapped];
            nativeBuildInputs = [pkgs.makeWrapper];
            postBuild = ''
              wrapProgram $out/bin/mcp-server-qdrant \
                --set ORT_DYLIB_PATH "${onnxruntime-bin}/lib/libonnxruntime${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}"
            '';
            inherit meta;
          };
      });

    devShells = forAllSystems ({
      pkgs,
      fenixPkgs,
      ...
    }: let
      toolchain = fenixPkgs.stable.withComponents [
        "cargo"
        "clippy"
        "rustc"
        "rustfmt"
        "rust-src"
        "rust-analyzer"
      ];
    in {
      default = pkgs.mkShell {
        packages =
          [
            toolchain
            pkgs.just
            pkgs.taplo
            pkgs.typos
            pkgs.actionlint
            pkgs.cargo-nextest
            pkgs.qdrant
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.apple-sdk_15
          ];

        env = {
          RUST_BACKTRACE = "1";
          RUST_SRC_PATH = "${toolchain}/lib/rustlib/src/rust/library";
        };
      };
    });
  };
}
