# author: ddukic

import evaluate
import torch

acccuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")
report = evaluate.load("bstrai/classification_report")


def compute_metrics_multitask(model, loader, task, eval_mode="macro"):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                ner_ids=batch["ner_ids"],
                task=task,
            )
            if task == "sentiment":
                target = batch["sentiment_labels"]
            else:
                target = batch["tone_labels"]
            pred = torch.argmax(out.logits, dim=-1)
            predictions.extend(pred.tolist())
            targets.extend(target.tolist())

    return {
        "accuracy": acccuracy.compute(predictions=predictions, references=targets)[
            "accuracy"
        ],
        "precision": precision.compute(
            predictions=predictions, references=targets, average=eval_mode
        )["precision"],
        "recall": recall.compute(
            predictions=predictions, references=targets, average=eval_mode
        )["recall"],
        "f1": f1.compute(
            predictions=predictions, references=targets, average=eval_mode
        )["f1"],
        "report": report.compute(predictions=predictions, references=targets),
    }


def logging_multitask(
    model,
    loader,
    loss,
    epoch,
    mode="train",
    eval_mode="macro",
):
    loss_string = mode + " loss: " + str(round(loss, 3))

    all_metrics_sentiment = compute_metrics_multitask(
        model=model,
        loader=loader,
        task="sentiment",
        eval_mode=eval_mode,
    )

    all_metrics_tone = compute_metrics_multitask(
        model=model,
        loader=loader,
        task="tone",
        eval_mode=eval_mode,
    )

    classification_string_sentiment = (
        "sentiment classification"
        + "\nAcc={:.3f}\tP={:.3f}\tR={:.3f}\tF1={:.3f}".format(
            all_metrics_sentiment["accuracy"],
            all_metrics_sentiment["precision"],
            all_metrics_sentiment["recall"],
            all_metrics_sentiment["f1"],
        )
    )

    classification_string_tone = (
        "tone classification"
        + "\nAcc={:.3f}\tP={:.3f}\tR={:.3f}\tF1={:.3f}".format(
            all_metrics_tone["accuracy"],
            all_metrics_tone["precision"],
            all_metrics_tone["recall"],
            all_metrics_tone["f1"],
        )
    )

    f1_string_sentiment = (
        mode + " sentiment f1: " + str(round(all_metrics_sentiment["f1"], 3))
    )
    f1_string_tone = mode + " tone f1: " + str(round(all_metrics_tone["f1"], 3))

    epoch_string = "\033[1m" + "Epoch: " + str(epoch) + "\033[0m"

    if mode == "train":
        print(epoch_string)
    print(loss_string)
    print(f1_string_sentiment)
    print(f1_string_tone)
    print("------------------------")
    print(mode + " evaluation details")
    print(classification_string_sentiment)
    print("------------------------")
    print(classification_string_tone)
    print("------------------------")

    return all_metrics_sentiment, all_metrics_tone
