import torch
import torch.nn as nn

from ignite.engine import Engine, Events
from ignite.metrics import Loss


def create_trainer(model, optimizer, criterion):
    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        X, y = batch
        preds = model(X).squeeze()

        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        return loss.item()
    
    return Engine(train_step)


def create_evaluator(model, criterion):
    def eval_step(engine, batch):
        model.eval()

        with torch.no_grad():
            X, y = batch
            preds = model(X).squeeze()

            return preds, y
        
        evaluator = Engine(eval_step)
        Loss(criterion).attach(evaluator, "mse")

        return evaluator


def run_training(model, training_loader, val_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    trainer = create_trainer(model, optimizer, criterion)
    evaluator = create_evaluator(model, criterion)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print(f"Epoch {engine.state.epoch} | Val Loss: {metrics["mse"]:.4f}")

    # Start training
    trainer.run(training_loader, max_epochs=epochs)
