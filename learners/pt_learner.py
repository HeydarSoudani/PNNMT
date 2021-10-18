import torch
import torch.nn.functional as F
import numpy as np
import time


def compute_prototypes(
    support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
  """
  Compute class prototypes from support features and labels
  Args:
    support_features: for each instance in the support set, its feature vector
    support_labels: for each instance in the support set, its label
  Returns:
    for each label of the support set, the average feature vector of instances with this label
  """
  seen_labels = torch.unique(support_labels)

  # Prototype i is the mean of all instances of features corresponding to labels == i
  return torch.cat(
    [
      support_features[(support_labels == l).nonzero(as_tuple=True)[0]].mean(0).reshape(1, -1)
      for l in seen_labels
    ]
  )


def pt_learner(model, queue, trg_queue, criterion, optim, args, device):
  model.train()  

  queue_length = len(queue)
  losses = 0

  for i in range(queue_length):
    optim.zero_grad()

    support_data = queue[i]["batch"]["support"]
    support_labels = support_data["label"].to(device)
    support_task = queue[i]["task"]

    _, support_features = model.forward(support_task, support_data)
    prototypes = compute_prototypes(support_features, support_labels)
    print('pt: {}'.format(prototypes.shape))

    # with torch.no_grad():
    query_data = trg_queue[0]["batch"]["query"]
    query_labels = query_data["label"].to(device)
    query_task = trg_queue[0]["task"]

    outputs, query_features = model.forward(query_task, query_data)
  
    loss = criterion(query_features, outputs, query_labels, prototypes)
    loss.backward()
    losses += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optim.step()

  return losses / queue_length


## For Pt.
def pt_evaluate(model, dataloader, prototypes, criterion, device):
  
  ce = torch.nn.CrossEntropyLoss()

  with torch.no_grad():
    total_loss = 0.0
    model.eval()
    for i, batch in enumerate(dataloader):

      sample, labels = batch
      sample, labels = sample.to(device), labels.to(device)
      
      logits, features = model.forward(sample)
      # loss = criterion(features, logits, labels, prototypes)
      loss = ce(logits, labels)
      # loss, acc = criterion(features, target=labels)
      loss = loss.mean()
      total_loss += loss.item()

    total_loss /= len(dataloader)
    return total_loss