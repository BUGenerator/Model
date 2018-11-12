# Model Generator: Code for training, etc

## Folder structure

- `tools/ENG_Grid`: environment setup for ENG Grid
- `python`: Code for the Deep Learning Model generation (still working on it; this is very basic yet)

## Model publishing

For every (usable) iteration of the model, we should...

1. Push appropriate commits to this repo (NEVER put the final model in Git);
2. Publish a new release and upload `.h5` model file to the release asset.

One-liner to download the latest model:

```shell
# model_fullres_keras.h5
$ curl -s https://api.github.com/repos/BUGenerator/Model/releases/latest | grep "browser_download_url.*model_fullres_keras.h5" | cut -d '"' -f 4 | wget -qi -
```