import torch
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.hadlers import ProgressBar


def create_trainer(model, optimizer, criterion, device="cpu", grad_clip=1.0, use_amp=True):
    # For mixed precision and faster inference
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device!="cpu"))

    def train_step(engine, batch):
        model.train()

        X, y = batch
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(use_amp and device != "cpu")):
            preds = model(X)
            preds = preds.view_as(y)


            loss = criterion(preds, y)
        
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        return loss.item()
    
    trainer = Engine(train_step)
    RunningAverage(output_transform= lambda x: x).attach(trainer, "loss")
    
    return trainer


def create_evaluator(model, criterion, device="cpu"):
    def eval_step(engine, batch):
        model.eval()

        with torch.no_grad():
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            preds = model(X)
            preds = preds.view_as(y)

            return preds, y
        
        evaluator = Engine(eval_step)
        Loss(criterion).attach(evaluator, "loss")

        return evaluator


def run_training(model, training_loader, val_loader, epochs=10, lr=1e-3, checkpoint_dir="checkpoints"):
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    trainer = create_trainer(model, optimizer, criterion, device=device)
    evaluator = create_evaluator(model, criterion, device=device)

    pbar = ProgressBar()
    pbar.attach(trainer, ["loss"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        evaluator.run(val_loader)
        
        train_loss = engine.state.metrics["loss"]
        val_loss = evaluator.state.metrics["loss"]

        print(
            f"Epoch [{engine.state.epoch}/{epochs}]"
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )
    
    checkpoint_handler = ModelCheckpoint(dirname=checkpoint_dir,
                                         filename_prefix="best",
                                         n_saved=2,
                                         require_empty=False,
                                         score_function=lambda engine: -engine.state.metrics["loss"],
                                         score_name="val_loss",
                                         global_step_transform=lambda *_: trainer.state.epoch)
    
    evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {"model": model})
    # Start training
    trainer.run(training_loader, max_epochs=epochs)

    return model
