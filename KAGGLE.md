# Running the notebook on Kaggle

You need a Kaggle account with a **verified phone number**. Without one, the GPU and Internet options are locked.

## 1. Create the notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **+ New Notebook**.
2. In the editor, choose **File → Import Notebook** and upload `landscape2art_cyclegan.ipynb` from this repo. You can also paste the GitHub URL of the file.
3. Give it a name, for example *Landscape2Art CycleGAN*, using the title at the top left.

## 2. Add the two datasets

In the right sidebar, open **Input → + Add Input**, search for each dataset and click **+** to add it:

| Search for | Dataset | Used as |
|---|---|---|
| `landscape pictures` | **arnaud58/landscape-pictures** | photos (domain X) |
| `wikiart` | **steubk/wikiart** | paintings (domain Y) |

You don't need to type any paths. The notebook finds both folders under `/kaggle/input` by itself.

## 3. Session settings

Open the right sidebar → **Session options** (or **Settings**):

- **Accelerator:** `GPU T4 x2` or `GPU P100`. Training uses one GPU, so either is fine.
- **Internet:** **On**. It's only needed to download the Inception-v3 weights for FID (~100 MB, once). Nothing is installed with `pip`. If you switch Internet on during a session, restart the session (**Run → Restart & clear outputs**) so the kernel actually gets network access.
- **Persistence:** leave as default.

## 4. Quick check (about 2 minutes, recommended)

Before starting the long run, check that everything is set up:

1. Run the cells from the top through **Data**.
2. The output should show `device: cuda (Tesla T4)` (or P100), `Internet: OK`, `FID: Inception-v3 weights ready.`, and a line like
   `Photos : /kaggle/input/... -> ~4250 train / 64 test` and `Art : .../Naive_Art_Primitivism -> ~2400 images`.
3. If a folder is not found, see Troubleshooting below.

Optional: set `EPOCHS = 1` and **Run All** to test the whole notebook in about 15 minutes. Set it back to `50` afterwards.

## 5. Start the real training run

Click **Save Version** (top right), choose **Save & Run All (Commit)**, then **Save**.

- The run happens **in the background**, so you can close the browser. Kaggle allows up to **12 hours** per run and about **30 GPU hours per week**.
- To follow progress, open **View Active Events** (bottom left) or the version's **Logs**. Each epoch prints one line with its losses and duration.
- The time for epoch 1 tells you whether all 50 epochs fit in one session. If they don't, the notebook stops safely at about 11.3 hours (`TIME_BUDGET_HOURS`) and keeps `checkpoints/last.pt`.

## 6. Resume if training didn't finish

1. Open the notebook in the editor again.
2. Go to **+ Add Input → Your Work → Notebooks**, pick **this notebook**, and add its latest output.
3. **Save Version → Save & Run All** again.

`RESUME_FROM = "auto"` finds `checkpoints/last.pt` from the previous output, but only if it was trained on the same `ART_STYLE`. Training then continues from the next epoch, with the optimiser state, LR schedule and loss history all restored.

## 6b. Only evaluate a finished run (no retraining)

Use this to compute FID or results for a run that has already trained, for example one where FID was skipped.

1. Make the checkpoint available. If `checkpoints/last.pt` is still in this session's **Output** panel, it's already there. Otherwise, attach that version's output as in step 6.
2. In the first cell, set `TRAIN = False`.
3. Click **Run All**.

The notebook loads the checkpoint and prints `TRAIN=False: skipping training ...`. It then runs only the evaluation cells: curves, gallery, cycle, FID and your own photos. This takes a few minutes. Running only the FID cell won't work after a restart, because it needs the models and data lists that the earlier cells create.

Set `TRAIN` back to `True` afterwards. If `TRAIN = False` and no checkpoint is found, the notebook stops with an error instead of evaluating an untrained model.

## 7. Get the results

Open the finished version → **Output** tab. You can download single files or everything at once:

- `generator_Naive_Art_Primitivism.pt`: final weights, all you need for inference
- `gallery.jpg`, `cycle.jpg`, `losses.png`, `metrics.json` (FID), `history.csv`
- `samples/epoch_XXX.jpg`: one preview per epoch, useful for choosing the best epoch visually
- `results/`: every held-out photo next to its painting

**Put your results in the README:** copy `gallery.jpg` and `losses.png` into `assets/`, then add the FID numbers from `metrics.json` to the Results section.

## 8. Try other styles

Change `ART_STYLE` in the first cell to any WikiArt folder name, for example `Cubism`, `Pop_Art`, `Ukiyo_e`, `Northern_Renaissance`, `Post_Impressionism` or `Pointillism`, then commit a new version. Every style gets its own checkpoints, and resume never mixes styles.

## 9. Stylize your own photos

Upload your photos as a dataset (**+ Add Input → Upload**), set `MY_PHOTOS_DIR = "/kaggle/input/<your-dataset>"` in section 11, and run that cell. Images are processed at up to 1024 px.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Landscape folder not found` / `Art folder ... not found` | Check that both datasets appear under *Input*. Otherwise set the paths yourself: `LANDSCAPE_DIR = "/kaggle/input/landscape-pictures"`, and `ART_DIR` to the style folder (copy the path from the Input panel with the copy icon). |
| `device: cpu` / the GPU warning appears | Set Accelerator to a GPU (step 3). If the GPU option is greyed out, verify your phone number. |
| `CUDA out of memory` | Set `BATCH_SIZE = 2`. |
| `Non-finite loss` | Set `USE_AMP = False`, or lower `LR` to `1e-4`. |
| `Internet: NOT reachable` / `FID will be skipped` | Training still runs; only FID is skipped. Check that Internet is on, then restart the session. If it still fails, download [`inception_v3_google-0cc3c7bd.pth`](https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth) on your computer, upload it as a Kaggle dataset (**+ Add Input → Upload**), and attach it. The notebook finds it automatically. |
| FID missing from a finished run | Follow step 6b (`TRAIN = False`, Run All). |
| Results look almost the same as the input photo | Train for more epochs; check that `LAMBDA_IDENTITY` is 0.5, not 5. |
| Gallery shows black-and-white photos | Keep `DROP_GRAYSCALE = True` (the default). The data cell prints how many it removed. |
| Run stopped at 12 h | Expected for long runs. Resume as described in step 6. |
