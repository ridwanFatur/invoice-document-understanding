import torch

def train_epoch(
    model, 
    optimizer, 
    train_loader, 
    device, 
):
    losses = []
    model.train()
    for idx, batch in enumerate(train_loader):
        image_tensors, input_ids, labels = batch
        
        image_tensors = image_tensors.to(device)
        input_ids = input_ids.to(device)
        labels = labels.to(device)  
        
        decoder_input_ids = input_ids[:, :-1]
        decoder_labels = labels[:, 1:]
        
        optimizer.zero_grad()
        
        decoder_outputs = model(
            image_tensors=image_tensors,
            decoder_input_ids=decoder_input_ids,
            decoder_labels=decoder_labels,
        )  
        loss = decoder_outputs['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()  
        losses.append(loss.item())
        
    return losses

def evaluate(model, test_loader, device, tokenizer):
    losses = []
    predicted_results = []

    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image_tensors, input_ids, labels = batch

            image_tensors = image_tensors.to(device)
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            decoder_input_ids = input_ids[:, :-1]
            decoder_labels = labels[:, 1:]

            decoder_outputs = model(
                image_tensors=image_tensors,
                decoder_input_ids=decoder_input_ids,
                decoder_labels=decoder_labels,
            )

            loss = decoder_outputs["loss"]
            losses.append(loss.item())

            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id

            ground_truth = tokenizer.batch_decode(
                labels_for_decode,
                skip_special_tokens=False,
            )

            prompts = ["<s_cord-v2>"] * image_tensors.size(0)

            result = model.predict_batch(
                image_tensors=image_tensors,
                tokenizer=tokenizer,
                prompts=prompts,
            )

            predictions = result["predictions"]

            for prediction, gt in zip(predictions, ground_truth):
                predicted_results.append({
                    "prediction": prediction,
                    "ground_truth": gt,
                })

    return losses, predicted_results
    