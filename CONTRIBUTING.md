
# Setup
We use uv as our dependency management, environment management, and build system.

After installing uv, use the following to init our project with all dependency groups.
```sh
uv sync --all-groups --all-extras
```

If you want to get the dependency of the cookbook
```sh
uv sync --all-packages --all-groups --all-extras
```

## Contribution Licensing

By submitting a pull request to the Standard ASR project, you agree to license your contribution under the project's Apache 2.0 License. You certify that you have the right to submit this contribution and that it does not violate any third-party rights. Your contribution will be attributed to you in the git history, and you will become part of "The Standard ASR Authors".
