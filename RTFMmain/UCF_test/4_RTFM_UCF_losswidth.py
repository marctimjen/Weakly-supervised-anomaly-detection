import neptune
import os
import numpy as np


token = os.getenv('NEPTUNE_API_TOKEN')
run = neptune.init_run(
    project="AAM/rtfmucf",
    api_token=token,
    with_id="RTFMUC-51"
)


test_loss = run["test/loss"].fetch_values()
test_loss = np.array(test_loss.value)
train_loss = run["train/loss"].fetch_values()
train_loss = np.array(train_loss.value)

print("Test-loss", np.mean(test_loss), "+/-", np.std(test_loss))
print("Train-loss", np.mean(train_loss), "+/-", np.std(train_loss))


run["test/loss_mean"] = np.mean(test_loss)
run["test/loss_std"] = np.std(test_loss)

run["train/loss_mean"] = np.mean(train_loss)
run["train/loss_std"] = np.std(train_loss)

run["model_pre"] = "nept_id_RTFMUC-38/rftm771-i3d.pkl"

run.stop()

