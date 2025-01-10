# UTAU Noise remover

-# scroll down for English.

## 日本語

これはUTAUのVB専用ではありませんが、`.wav`ファイルを全て再帰的に検索し、全ておきかえるというものです。

`python --input-dir <UTAU VB PATH>` とだけやれば、あとは適当にやってくれます

### パラメータ

`--input-dir` ボイスバンクの場所。処理する`.wav`ファイルが含まれているディレクトリを指定します。
`--output-dir` ボイスバンクデータが消えるのが怖いならこれ。処理後のファイルを保存するディレクトリを指定します。
`--max-workers` ノイズ除去プログラムがどれだけ並列でおこなわれるか(デフォルト5)。並列処理のための最大ワーカー数を指定します。GPUの筋力に自信ある人はどうぞ。
`-v` `--verbose` デバッグ用。詳細なログ出力を有効にします。
`--leave-artifacts` 主にデバッグ用ですが、ファイルののこりカスとかみたいならどうぞ。プログラムがだしたゴミ片付けません。処理後の一時ファイルを残します。
`--model` 使用するモデル名 (デフォルトは "mdx_extra_q")。ノイズ除去に使用するモデルを指定します。
`--no-cuda` CUDAを無効にする。GPUを使用せずにCPUのみで処理を行います。

### 注意

- [!] 絶対にファイルのコピーを保存しておいてください！プログラムで変更かけられたファイルは元に戻りません！

このプログラムはCUDAを使用してGPUでの高速処理をサポートしています。CUDAを有効にするには、適切なバージョンのPyTorchとCUDAがインストールされている必要があります。詳細は[PyTorchの公式ドキュメント](https://pytorch.org/get-started/locally/)を参照してください。

## English

This is not exclusively for UTAU's VB, but it recursively searches for all `.wav` files and replaces them all.

Just run `python --input-dir <UTAU VB PATH>` and it will take care of the rest.

### Parameters

`--input-dir` Location of the voice bank. Specify the directory containing the `.wav` files to be processed.
`--output-dir` If you're afraid of losing voice bank data, use this. Specify the directory where the processed files will be saved.
`--max-workers` How many noise removal programs run in parallel (default is 5). Specify the maximum number of workers for parallel processing. If your GPU is strong, increase the number.
`-v` `--verbose` For debugging. Enable verbose logging.
`--leave-artifacts` Mainly for debugging, but if you want to see the leftover files, use this. The program won't clean up the garbage it produces. Leave temporary files after processing.
`--model` Model name to use for processing (default is "mdx_extra_q"). Specify the model to use for noise removal.
`--no-cuda` Disable CUDA. Process using the CPU only, without using the GPU.

### Note

- [!] be sure to ALWAYS have a backup of the voicebank. The operation is irreversible.

This program supports fast processing on the GPU using CUDA. To enable CUDA, you need to have the appropriate version of PyTorch and CUDA installed. For more details, refer to the [official PyTorch documentation](https://pytorch.org/get-started/locally/).
