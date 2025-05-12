# main.py
import os.path
import warnings

warnings.filterwarnings("ignore")
from coco_eval import model_evaluate, coco_caption_eval, inference
from torch import optim, nn

# import tmp.diffcap_eval as diffcap_eval
from diff_models.diffcap_model import Diffuser, Diffuser_with_LN
from my_utils.blip_util import load_checkpoint
from diff_models.diffusion import *

# from dataload.dataloader import train_loader, val_loader, val_set
from torch.utils.data import DataLoader
from dataload import create_dataset
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
from my_utils.train_util import batch_loss
import time
from transformers import get_linear_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from my_utils.detr_object import get_detr_objects
import wandb
import infer

wandb_configs = {
    "epochs": EPOCH_NUM,
    "batch_size": TRAIN_BATCH_SIZE,
    "length": MAX_LENGTH,
}

if running_in_ipython_family():

    accelerator.init_trackers(
        "Diff-Cap",
        config=wandb_configs,
        init_kwargs={"wandb": {"mode": "disabled"}},  # Disable wandb logging
    )
else:
    accelerator.init_trackers(
        "Diff-Cap",
        config=wandb_configs,
        init_kwargs={"wandb": {"name": notes, "entity": "tlebryk-harvard-university"}},
    )  # , "entity": "xxx"}})
data_config = {
    "image_size": 224,
    "ann_root": "datasets/COCO/",
    "image_root": "datasets/COCO/web/all_data",
}
train_set, val_set, test_set = create_dataset("caption_coco", data_config)
print(len(train_set), len(val_set), len(test_set))
try:
    print(f"{train_set[0]}, {val_set[0]}")
except Exception as e:
    print(e)
train_loader = DataLoader(
    train_set, shuffle=True, batch_size=TRAIN_BATCH_SIZE, drop_last=True, num_workers=32
)
val_loader = DataLoader(
    val_set, shuffle=False, batch_size=VAL_BATCH_SIZE, drop_last=True, num_workers=2
)

test_loader = DataLoader(
    test_set, shuffle=False, batch_size=VAL_BATCH_SIZE, drop_last=True, num_workers=2
)
PRETRAINED_DIR = "pretrained_ckpt"


def build_model():
    model = Diffuser_with_LN(image_size=224)
    model.visual_encoder, _ = load_checkpoint(
        model.visual_encoder, f"{PRETRAINED_DIR}/model_base_capfilt_large.pth"
    )
    if FULL_MODEL_PATH:
        state = torch.load(FULL_MODEL_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
    else:
        # final_model_dir = f"{LOG_DIR}/{MODEL_NAME}"
        # state = torch.load(
        #     os.path.join(final_model_dir, "pytorch_model.bin"), map_location=device
        # )
        state = torch.load("pytorch_model.bin", map_location=device)
    return model  # .to(device).eval()


# if not USING_TIME_LN:
#     model = Diffuser(image_size=224)
# else:
model = build_model()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
)
num_step = len(train_loader) * EPOCH_NUM
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_step * WARMUP_RATIO, num_training_steps=num_step
)
# scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_step * WARMUP_RATIO)
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_step * WARMUP_RATIO,
#                                            num_training_steps=num_step, num_cycles=2)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
special = tokenizer(["."], return_tensors="pt")
special_emb = model.space_encoder(special["input_ids"])[0][0][1]
import pandas as pd
import os
from datetime import datetime

# Create a logs directory if it doesn't exist
log_dir = f"{LOG_DIR}/{MODEL_NAME}/logs"
os.makedirs(log_dir, exist_ok=True)

# Initialize empty log DataFrames
train_logs = pd.DataFrame()
val_logs = pd.DataFrame()


# Helper function to safely extract value from tensor or use directly if already a number
def safe_item(value):
    if hasattr(value, "item"):
        return value.item()
    return value


# Function to log metrics to CSV
def log_to_csv(metrics, step, is_validation=False):
    global train_logs, val_logs

    # Add step information and convert tensors to python values
    metrics_with_step = {k: safe_item(v) for k, v in metrics.items()}
    metrics_with_step["step"] = step

    # Convert to DataFrame (single row)
    metrics_df = pd.DataFrame([metrics_with_step])

    # Append to the appropriate log DataFrame
    if is_validation:
        val_logs = pd.concat([val_logs, metrics_df], ignore_index=True)
        val_logs.to_csv(f"{log_dir}/validation_logs.csv", index=False)
        print(f"Validation metrics saved at step {step}")
    else:
        train_logs = pd.concat([train_logs, metrics_df], ignore_index=True)
        train_logs.to_csv(f"{log_dir}/training_logs.csv", index=False)
        if step % 50 == 0:  # Don't print too often
            print(f"Training metrics saved at step {step}")


# Initialize trackers but disable wandb


def train_func(model, trainer, x, scheduler, train=True):
    x_0 = model.space_encoder(
        input_ids=x["input_ids"], attention_mask=x["attention_mask"]
    )[0]
    # torch.save(torch.mean(x_0, dim=(0,1)), 'datasets/mean_emb_split.pickle')
    # torch.save(torch.sqrt(torch.var(x_0, dim=(0, 1))), 'datasets/std_emb_split.pickle')

    x_0 = (x_0 - X_MEAN.to(accelerator.device)) / X_SIGMA.to(accelerator.device)
    atten_mask = x["attention_mask"]
    atten_mask = torch.roll(atten_mask, -1, 1)
    atten_mask[:, 0] = 0
    atten_mask[:, -1] = 0
    # change pad cls sep to special token
    x_0[atten_mask == 0] = special_emb.to(accelerator.device)
    if USE_OBJECT:
        # objects_list, objects_ids, objects_mask = get_detr_objects(x['detr_input'])
        # objects_ids, objects_mask = objects_ids.to(accelerator.device), objects_mask.to(accelerator.device)
        objects_ids, objects_mask = x["objects_ids"].to(accelerator.device), x[
            "objects_mask"
        ].to(accelerator.device)
        object_emb = model.space_encoder(
            input_ids=objects_ids.to(accelerator.device),
            attention_mask=objects_mask.to(accelerator.device),
        )[0]
        object_emb = (object_emb - X_MEAN.to(accelerator.device)) / X_SIGMA.to(
            accelerator.device
        )
        object_atten_mask = objects_mask
        object_atten_mask = torch.roll(object_atten_mask, -1, 1)
        object_atten_mask[:, 0] = 0
        object_atten_mask[:, -1] = 0
        object_emb[object_atten_mask == 0] = special_emb.to(accelerator.device)

        # randomly mask some objects
        object_rand_mask = torch.rand(object_emb.shape[0]) < OBJECT_MASK_RATIO
        object_emb[object_rand_mask == 1] = repeat(
            special_emb, "d -> seq d", seq=object_emb.shape[1]
        ).to(accelerator.device)

    t = torch.randint(
        0, STEP_TOT, (x_0.shape[0],), device=accelerator.device
    )  # 随机采样batchsize个时间
    if X_0_PREDICTION or EPSILON_PRED:
        x_t = diffuse_t(x_0, t)  # bsz, seqlen, dmodel
        x_tgt = None
    # else:
    #     x_t, x_tgt = generate_diffuse_pair(x_0, t, torch.max(t - X_T_STEP_INTERVAL,
    #                                                          torch.zeros(t.shape, device=accelerator.device,
    #                                                                      dtype=torch.int64)))
    x_1 = diffuse_t(x_0, torch.ones(1, dtype=torch.int64, device=accelerator.device))
    image, mask = x["image"].to(accelerator.device), x["attention_mask"].to(
        accelerator.device
    )
    if CLASSIFIER_FREE_PROB > 0 and train:
        if train:
            train_batch_size = TRAIN_BATCH_SIZE
        else:
            train_batch_size = VAL_BATCH_SIZE
        classifier_mask = (
            (torch.rand(train_batch_size) > CLASSIFIER_FREE_PROB)
            .type(torch.float32)
            .to(accelerator.device)
        )  # generate mask
        image = image * (
            repeat(
                classifier_mask, "b -> b c h w", c=3, h=image.shape[2], w=image.shape[3]
            )
        )
    x_pred = torch.zeros_like(x_t)
    if USE_OBJECT:
        object_emb_selfcond = torch.concat([object_emb, object_emb], dim=-1)
        # add self attentioning
        if SELF_COND and random.random() > SELF_COND_PROB:
            concat_x_t = torch.cat([x_t, x_pred], dim=-1)
            concat_x_t = torch.cat([concat_x_t, object_emb_selfcond], dim=-2)
            x_pred = model(
                image, concat_x_t, torch.concat([mask, objects_mask], dim=-1), t
            )
            x_pred = x_pred.detach()
        # x_t restore loss
        x_pred = model(
            image,
            torch.concat(
                [torch.cat([x_t, x_pred], dim=-1), object_emb_selfcond], dim=-2
            ),
            torch.concat([mask, objects_mask], dim=-1),
            t,
        )
    else:
        if SELF_COND and random.random() > SELF_COND_PROB:
            concat_x_t = torch.cat([x_t, x_pred], dim=-1)
            # concat_x_t = torch.cat([concat_x_t, object_emb_selfcond], dim=-2)
            x_pred = model(image, concat_x_t, mask, t)
            x_pred = x_pred.detach()
        # x_t restore loss
        x_pred = model(image, torch.cat([x_t, x_pred], dim=-1), mask, t)

    x_t_loss, x_1_loss, prob_loss, valid_token_loss, pad_loss = batch_loss(
        model, x_pred, x_t, x_tgt, x_0, x["attention_mask"], x["input_ids"], LOSS_FUNC
    )

    l = x_t_loss + x_1_loss + prob_loss

    if train:
        trainer.zero_grad()
        accelerator.backward(l)
        # accelerator.clip_grad_norm_(model.parameters(), 1.0)
        trainer.step()
        scheduler.step()

    return l, x_t_loss, x_1_loss, prob_loss, valid_token_loss, pad_loss


def validate(model, epoch=None):
    val_acc_x_t = 0
    val_acc_x_1 = 0
    val_acc_prob = 0
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_num, x in tqdm(
            enumerate(val_loader), disable=not accelerator.is_local_main_process
        ):
            # try:
            if batch_num == 0:
                z = inference(x, tokenizer, model)

                # Save z to a file if epoch is specified
                if epoch is not None and accelerator.is_local_main_process:
                    # Create directories if they don't exist
                    os.makedirs(f"{LOG_DIR}/{MODEL_NAME}/inferences", exist_ok=True)

                    # Save inference output to file
                    z_filename = (
                        f"{LOG_DIR}/{MODEL_NAME}/inferences/epoch_{epoch}_inference.txt"
                    )
                    with open(z_filename, "w", encoding="utf-8") as f:
                        # Ensure z is a string
                        if not isinstance(z, str):
                            z_str = str(z)
                        else:
                            z_str = z
                        f.write(z_str)

                    # Save x['text'] to a file if it exists
                    if "text" in x:
                        os.makedirs(f"{LOG_DIR}/{MODEL_NAME}/texts", exist_ok=True)
                        text_filename = (
                            f"{LOG_DIR}/{MODEL_NAME}/texts/epoch_{epoch}_text.txt"
                        )
                        with open(text_filename, "w", encoding="utf-8") as f:
                            # Handle different formats of x['text']
                            if isinstance(x["text"], list):
                                text_str = "\n".join(x["text"])
                            else:
                                text_str = str(x["text"])
                            f.write(text_str)

                print(z)
            l, x_t_loss, x_1_loss, prob_loss, valid_token_loss, pad_loss = train_func(
                model, optimizer, x, scheduler, train=False
            )
            val_acc_x_t += x_t_loss
            val_acc_x_1 += x_1_loss
            val_acc_prob += prob_loss
            val_loss += l
        # except Exception as e:
        #     print(e)
    model.train()

    return (
        val_loss / len(val_loader),
        val_acc_x_t / len(val_loader),
        val_acc_x_1 / len(val_loader),
        val_acc_prob / len(val_loader),
    )


model, optimizer, train_loader, val_loader, scheduler, X_MEAN, X_SIGMA = (
    accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler, X_MEAN, X_SIGMA
    )
)

if START_EPOCH > 0:
    accelerator.load_state(f"{LOG_DIR}/{MODEL_NAME}/acc_epoch_{START_EPOCH}/")
if isinstance(model, DistributedDataParallel):
    model = model.module
# early_stopped = False


######################################################################################
#################### begin training #################################################
######################################################################################

if not os.path.exists(f"{LOG_DIR}/{MODEL_NAME}"):
    os.makedirs(f"{LOG_DIR}/{MODEL_NAME}", exist_ok=True)


accelerator.print("start training")
start_time = time.time()
start_epoch = START_EPOCH
accelerator.print("after sync", (time.time() - start_time) / 60, "min")
(
    l,
    x_t_loss,
    x_1_loss,
    prob_loss,
) = validate(model)
accelerator.log(
    {
        "val_loss": l,
        "val_x_t_loss": x_t_loss,
        "val_x_1_loss": x_1_loss,
        "val_prob_loss": prob_loss,
        #  'val_valid_token_loss': valid_token_loss,
        #  'val_pad_loss': pad_loss
    }
)

model.train()


for epoch in range(start_epoch, EPOCH_NUM):
    accelerator.print(f"current epoch{epoch}")
    acc_x_t = 0
    acc_x_1 = 0
    acc_prob = 0
    acc_l = 0

    accelerator.print("the number of batchs is", len(train_loader))
    accelerator.print("before training", (time.time() - start_time) / 60, "min")
    for batch_num, x in tqdm(
        enumerate(train_loader), disable=not accelerator.is_local_main_process
    ):
        l, x_t_loss, x_1_loss, prob_loss, valid_token_loss, pad_loss = train_func(
            model, optimizer, x, scheduler
        )
        if batch_num % 50 == 0:
            accelerator.log(
                {
                    "loss": l,
                    "x_t_loss": x_t_loss,
                    "x_1_loss": x_1_loss,
                    "prob_loss": prob_loss,
                    "valid_token_loss": valid_token_loss,
                    "pad_loss": pad_loss,
                }
            )
            log_metrics = {
                "loss": l,
                "x_t_loss": x_t_loss,
                "x_1_loss": x_1_loss,
                "prob_loss": prob_loss,
                "valid_token_loss": valid_token_loss,
                "pad_loss": pad_loss,
                "epoch": epoch,
                "batch": batch_num,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            log_to_csv(log_metrics, epoch * len(train_loader) + batch_num)

        acc_x_t += x_t_loss
        acc_x_1 += x_1_loss
        acc_prob += prob_loss
        acc_l += l

        if DYNAMIC_ROUNDING_WEIGHT > 0:
            ROUNDING_WEIGHT = (
                (acc_x_t + acc_x_1) / acc_prob
            ).detach() * DYNAMIC_ROUNDING_WEIGHT

        if DEBUG:
            break
    accelerator.print("after a epoch training", (time.time() - start_time) / 60, "min")
    accelerator.wait_for_everyone()
    (
        l,
        x_t_loss,
        x_1_loss,
        prob_loss,
    ) = validate(model)
    # After validation, replace accelerator.log with:
    val_metrics = {
        "val_loss": l,
        "val_x_t_loss": x_t_loss,
        "val_x_1_loss": x_1_loss,
        "val_prob_loss": prob_loss,
        "epoch": epoch,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_to_csv(val_metrics, epoch * len(train_loader), is_validation=True)
    accelerator.log(val_metrics)
    accelerator.print(val_metrics)

    # Upload artifacts at the end of each epoch
    if accelerator.is_local_main_process:
        wandb_run = accelerator.get_tracker("wandb", unwrap=True)

        # Upload inference artifact
        z_filename = f"{LOG_DIR}/{MODEL_NAME}/inferences/epoch_{epoch}_inference.txt"
        if os.path.exists(z_filename):
            z_artifact = wandb.Artifact(f"inference_epoch_{epoch}", type="text")
            z_artifact.add_file(z_filename)
            wandb_run.log_artifact(z_artifact)

        # Upload text artifact
        text_filename = f"{LOG_DIR}/{MODEL_NAME}/texts/epoch_{epoch}_text.txt"
        if os.path.exists(text_filename):
            text_artifact = wandb.Artifact(f"text_epoch_{epoch}", type="text")
            text_artifact.add_file(text_filename)
            wandb_run.log_artifact(text_artifact)

    # unwrapped_model = accelerator.unwrap_model(model)
    # accelerator.save(unwrapped_model.state_dict(), f"./checkpoint/{MODEL_NAME}/epoch_{epoch}.pickle")
    # model = model.to(accelerator.device)
    if epoch % SAVE_EPOCHS == 0:
        accelerator.save_state(f"{LOG_DIR}/{MODEL_NAME}/acc_epoch_{epoch}/")
        accelerator.print("after saving", (time.time() - start_time) / 60, "min")
        # Also save to wandb
        if accelerator.is_local_main_process:
            # Get the unwrapped model
            unwrapped_model = accelerator.unwrap_model(model)

            # Save model checkpoint for wandb
            checkpoint_path = (
                f"{LOG_DIR}/{MODEL_NAME}/acc_epoch_{epoch}/pytorch_model.bin"
            )
            torch.save(unwrapped_model.state_dict(), checkpoint_path)

            # Log to wandb as an artifact
            wandb_run = accelerator.get_tracker("wandb", unwrap=True)
            artifact = wandb.Artifact(f"diff-cap-ckpt-epoch-{epoch}", type="checkpoint")
            artifact.add_file(checkpoint_path)
            wandb_run.log_artifact(artifact)

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
if accelerator.is_local_main_process:
    final_model_dir = f"{LOG_DIR}/{MODEL_NAME}"
    os.makedirs(final_model_dir, exist_ok=True)
    # Save the model state dict as pytorch_model.bin
    torch.save(
        unwrapped_model.state_dict(), os.path.join(final_model_dir, "pytorch_model.bin")
    )
    wandb_run = accelerator.get_tracker(
        "wandb", unwrap=True
    )  # grab the underlying wandb Run
    artifact = wandb.Artifact("diff-cap-ckpt", type="model")
    artifact.add_file(os.path.join(final_model_dir, "pytorch_model.bin"))
    wandb_run.log_artifact(artifact)  # upload once per epoch

    # Optionally, if it's a HuggingFace model and has a config, you can save that too
    # if hasattr(unwrapped_model, 'config'):
    #     unwrapped_model.config.to_json_file(os.path.join(final_model_dir, "config.json"))

    accelerator.print(f"Model saved to {final_model_dir}/pytorch_model.bin")

# unwrapped_model = accelerator.unwrap_model(model)
# accelerator.save(unwrapped_model.state_dict(), f"./checkpoint/{MODEL_NAME}.pickle")
# model = model.to(accelerator.device)
accelerator.print("Done!")
if accelerator.is_local_main_process:
    results, metrics = infer.run_inference_on_dataset(model, val_loader, tokenizer)
    accelerator.log(metrics)

    infer.save_results(results, metrics, f"result/{RESULT_FILE}.json")
    artifact = wandb.Artifact("diff-cap-results", type="results")
    artifact.add_file(f"result/{RESULT_FILE}.json")
    wandb_run.log_artifact(artifact)  # upload once per epoch

    # bleu = diffcap_eval.evaluate(model, val_set, val_loader)
    # accelerator.log({"bleu": bleu})
    # model_evaluate(model, val_set, val_loader)
    # if not os.path.exists("result"):
    #     os.makedirs("result", exist_ok=True)
    # coco_caption_eval("result/", f"result/{RESULT_FILE}.json", split="val")
accelerator.end_training()
