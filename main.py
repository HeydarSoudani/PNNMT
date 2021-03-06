import argparse, time, torch, os, logging, warnings, sys
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from data import CorpusQA, CorpusSC, CorpusTC, CorpusPO, CorpusPA
from model import BertMetaLearning
from datapath import loc, get_loc

from losses import CPELoss


# from samplers.reptile_sampler import TaskSampler
# from learners.reptile_learner import reptile_learner
from samplers.pt_sampler import TaskSampler
from learners.pt_learner import pt_learner


from utils.logger import Logger

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--meta_lr", type=float, default=2e-5, help="meta learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="")
parser.add_argument("--hidden_dims", type=int, default=768, help="")  # 768

parser.add_argument("--lambda_1", type=float, default=1.0, help="DCE Coefficient in loss function")
parser.add_argument("--lambda_2", type=float, default=0.5, help="CE Coefficient in loss function")
parser.add_argument("--lambda_3", type=float, default=0.001, help="PL Coefficient in loss function")
parser.add_argument("--temp_scale", type=float, default=0.2, help="Temperature scale for DCE in loss function")

# bert-base-multilingual-cased
# xlm-roberta-base
parser.add_argument(
  "--model_name",
  type=str,
  default="xlm-roberta-base",
  help="name of the pretrained model",
)
parser.add_argument(
  "--local_model", action="store_true", help="use local pretrained model"
)

parser.add_argument("--sc_labels", type=int, default=3, help="")
parser.add_argument("--qa_labels", type=int, default=2, help="")
parser.add_argument("--tc_labels", type=int, default=10, help="")
parser.add_argument("--po_labels", type=int, default=18, help="")
parser.add_argument("--pa_labels", type=int, default=2, help="")

parser.add_argument("--qa_batch_size", type=int, default=8, help="batch size")
parser.add_argument("--sc_batch_size", type=int, default=32, help="batch size")  # 32
parser.add_argument("--tc_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--po_batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--pa_batch_size", type=int, default=8, help="batch size")

parser.add_argument("--task_per_queue", type=int, default=8, help="")
parser.add_argument("--update_step", type=int, default=3, help="number of REPTILE update steps")
parser.add_argument("--beta", type=float, default=1.0, help="")

# ---------------
parser.add_argument("--epochs", type=int, default=5, help="iterations")  # 5
parser.add_argument("--start_epoch", type=int, default=0, help="start iterations from")  # 0
parser.add_argument("--ways", type=int, default=3, help="number of ways")  # 2
parser.add_argument("--shot", type=int, default=4, help="number of shots")  # 4
parser.add_argument("--query_num", type=int, default=4, help="number of queries")  # 0
parser.add_argument("--meta_iteration", type=int, default=3000, help="")
# ---------------

parser.add_argument("--seed", type=int, default=42, help="seed for numpy and pytorch")
parser.add_argument(
    "--log_interval",
    type=int,
    default=200,
    help="Print after every log_interval batches",
)
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--tpu", action="store_true", help="use TPU")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="", help="")
parser.add_argument("--log_file", type=str, default="main_output.txt", help="")
parser.add_argument("--grad_clip", type=float, default=5.0)
parser.add_argument("--meta_tasks", type=str, default="sc,pa,qa,tc,po")
parser.add_argument("--target_task", type=str, default="sc_fa")

parser.add_argument("--sampler", type=str, default="uniform_batch", choices=["uniform_batch"])
parser.add_argument("--temp", type=float, default=1.0)

parser.add_argument("--num_workers", type=int, default=0, help="")
parser.add_argument("--n_best_size", default=20, type=int)  # 20
parser.add_argument("--max_answer_length", default=30, type=int)  # 30
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
)
parser.add_argument("--scheduler", action="store_true", help="use scheduler")
parser.add_argument("--step_size", default=3000, type=int)
parser.add_argument("--last_step", default=0, type=int)
parser.add_argument("--gamma", default=0.1, type=float)
parser.add_argument("--warmup", default=0, type=int)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.save):
  os.makedirs(args.save)

sys.stdout = Logger(os.path.join(args.save, args.log_file))
print(args)

task_types = args.meta_tasks.split(",")
list_of_tasks = []

for tt in loc["train"].keys():
  if tt[:2] in task_types:
    list_of_tasks.append(tt)

for tt in task_types:
  if "_" in tt:
    list_of_tasks.append(tt)

list_of_tasks = list(set(list_of_tasks))
print(list_of_tasks)


def evaluate(model, data, device):
  with torch.no_grad():
    total_loss = 0.0
    for batch in data:
      data_labels = batch["label"].to(device)
      output, _ = model.forward(args.target_task, batch)
      loss = F.cross_entropy(output, data_labels, reduction="none")
      loss = loss.mean()
      total_loss += loss.item()
    total_loss /= len(data)
    return total_loss


def evaluateMeta(model, dev_loaders, device):
  model.eval()
  loss = evaluate(model, dev_loaders[0], device)
  return loss


def main():
  ### == Device ======================
  if torch.cuda.is_available():
    if not args.cuda:
      args.cuda = True
    torch.cuda.manual_seed_all(args.seed)
  DEVICE = torch.device("cuda" if args.cuda else "cpu")


  ### == target dataset ==============
  k = args.target_task
  trg_train_corpus = None
  trg_dev_corpus = None
  trg_batch_size = 32
  if "qa" in k:
    trg_train_corpus = CorpusQA(
      *get_loc("train", k, args.data_dir),
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_dev_corpus = CorpusQA(
      *get_loc("dev", k, args.data_dir),
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_batch_size = args.qa_batch_size
  elif "sc" in k:
    trg_train_corpus = CorpusSC(
      *get_loc("train", k, args.data_dir),
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_dev_corpus = CorpusSC(
      *get_loc("dev", k, args.data_dir),
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_batch_size = args.sc_batch_size
  elif "tc" in k:
    trg_train_corpus = CorpusTC(
      get_loc("train", k, args.data_dir)[0],
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_dev_corpus = CorpusTC(
      get_loc("dev", k, args.data_dir)[0],
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_batch_size = args.tc_batch_size
  elif "po" in k:
    trg_train_corpus = CorpusPO(
      get_loc("train", k, args.data_dir)[0],
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_dev_corpus = CorpusPO(
      get_loc("dev", k, args.data_dir)[0],
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_batch_size = args.po_batch_size
  elif "pa" in k:
    trg_train_corpus = CorpusPA(
      get_loc("train", k, args.data_dir)[0],
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_dev_corpus = CorpusPA(
      get_loc("dev", k, args.data_dir)[0],
      model_name=args.model_name,
      local_files_only=args.local_model,
    )
    trg_batch_size = args.pa_batch_size

  trg_train_sampler = TaskSampler(
    trg_train_corpus,
    n_way=args.ways,
    n_shot=0,
     # n_query_way=args.query_ways,
    n_query=args.query_num,
    n_tasks=args.meta_iteration,
  )
  trg_train_loader = DataLoader(
    trg_train_corpus,
    batch_sampler=trg_train_sampler,
    num_workers=args.num_workers,
    pin_memory=True,
    collate_fn=trg_train_sampler.episodic_collate_fn,
  )
  trg_dev_loader = DataLoader(
    trg_dev_corpus,
    batch_size=trg_batch_size,
    pin_memory=True
  )

  ### ================================
  
  ### == auxiliary loader ============
  train_loaders = []
  dev_loaders = []

  for k in list_of_tasks:
    train_corpus = None
    dev_corpus = None
    batch_size = 32

    if "qa" in k:
      train_corpus = CorpusQA(
        *get_loc("train", k, args.data_dir),
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      dev_corpus = CorpusQA(
        *get_loc("dev", k, args.data_dir),
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      batch_size = args.qa_batch_size
    elif "sc" in k:
      train_corpus = CorpusSC(
        *get_loc("train", k, args.data_dir),
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      dev_corpus = CorpusSC(
        *get_loc("dev", k, args.data_dir),
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      batch_size = args.sc_batch_size
    elif "tc" in k:
      train_corpus = CorpusTC(
        get_loc("train", k, args.data_dir)[0],
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      dev_corpus = CorpusTC(
        get_loc("dev", k, args.data_dir)[0],
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      batch_size = args.tc_batch_size
    elif "po" in k:
      train_corpus = CorpusPO(
        get_loc("train", k, args.data_dir)[0],
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      dev_corpus = CorpusPO(
        get_loc("dev", k, args.data_dir)[0],
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      batch_size = args.po_batch_size
    elif "pa" in k:
      train_corpus = CorpusPA(
        get_loc("train", k, args.data_dir)[0],
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      dev_corpus = CorpusPA(
        get_loc("dev", k, args.data_dir)[0],
        model_name=args.model_name,
        local_files_only=args.local_model,
      )
      batch_size = args.pa_batch_size
    else:
      continue

    train_sampler = TaskSampler(
      train_corpus,
      n_way=args.ways,
      n_shot=args.shot,
      n_query=0,
      n_tasks=args.meta_iteration,
    )
    train_loader = DataLoader(
      train_corpus,
      batch_sampler=train_sampler,
      num_workers=args.num_workers,
      pin_memory=True,
      collate_fn=train_sampler.episodic_collate_fn,
    )
    train_loaders.append(train_loader)

    dev_loader = DataLoader(
      dev_corpus,
      batch_size=batch_size,
      pin_memory=True
    )
    dev_loaders.append(dev_loader)

  ### ================================

  ### == Model =======================
  model = BertMetaLearning(args).to(DEVICE)
  if args.load != "":
    print(f"loading model {args.load}...")
    model = torch.load(args.load)

  # steps = args.epochs * args.meta_iteration

  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [
        p
        for n, p in model.named_parameters()
        if not any(nd in n for nd in no_decay)
      ],
      "weight_decay": args.weight_decay,
      "lr": args.meta_lr,
    },
    {
      "params": [
        p
        for n, p in model.named_parameters()
        if any(nd in n for nd in no_decay)
      ],
      "weight_decay": 0.0,
      "lr": args.meta_lr,
    },
  ]

  optim = AdamW(
    optimizer_grouped_parameters,
    lr=args.meta_lr,
    eps=args.adam_epsilon
  )

  scheduler = StepLR(
    optim,
    step_size=args.step_size,
    gamma=args.gamma,
    last_epoch=args.last_step - 1,
  )
  criterion = CPELoss(args)

  logger = {}
  logger["total_val_loss"] = []
  logger["val_loss"] = {k: [] for k in list_of_tasks}
  logger["train_loss"] = []
  logger["args"] = args


  ## == training ======
  global_time = time.time()

  min_loss = float("inf")
  min_task_losses = {
    "qa": float("inf"),
    "sc": float("inf"),
    "po": float("inf"),
    "tc": float("inf"),
    "pa": float("inf"),
  }

  try:
    for epoch_item in range(args.start_epoch, args.epochs):
      print(
        "===================================== Epoch %d ====================================="
        % epoch_item
      )
      train_loss = 0.0

      train_loader_iterations = [
        iter(train_loader) for train_loader in train_loaders
      ]
      trg_train_loader_iteration = iter(trg_train_loader)
      for miteration_item in range(args.meta_iteration):

        # == Data preparation ===========
        queue = [
          {"batch": next(train_loader_iterations[i]), "task": task}
          for i, task in enumerate(list_of_tasks)
        ]
        trg_queue = [{"batch": next(trg_train_loader_iteration), "task": args.target_task}]
      
        ## == train ===================
        # loss = reptile_learner(model, queue, optim, miteration_item, args)
        loss = pt_learner(
          model,
          queue,
          trg_queue,
          criterion,
          optim,
          args,
          device=DEVICE)
        train_loss += loss

        ## == validation ==============
        if (miteration_item + 1) % args.log_interval == 0:
          total_loss = train_loss / args.log_interval
          train_loss = 0.0

          # evalute on val_dataset
          val_loss_total = evaluateMeta(model, [trg_dev_loader], device=DEVICE)
          print(
            "Time: %f, Step: %d, Train Loss: %f, Val Loss: %f"
            % (
              time.time() - global_time,
              miteration_item + 1,
              total_loss,
              val_loss_total,
            )
          )
          global_time = time.time()

          if val_loss_total < min_loss:
            torch.save(
              model, os.path.join(args.save, "model_" + args.target_task + ".pt"),
            )
            min_loss = val_loss_total
            print("Saving " + args.target_task + "  Model")
          total_loss = 0
          print("===============================================")
        if args.scheduler:
          scheduler.step()

  except KeyboardInterrupt:
    print("skipping training")

  # save last model
  torch.save(model, os.path.join(args.save, "model_last.pt"))
  print("Saving new last model")


if __name__ == "__main__":
    main()

