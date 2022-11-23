#
# def epoch(epoch_index, tb_writer):
#     running_loss = 0.0
#     last_loss = 0.0
#     for index, data in enumerate(train_loader):
#         inputs, labels = data
#         inputs = inputs.to(device, torch.float)
#         labels = labels.to(device, torch.float)
#
#         optimizer.zero_grad()
#
#         outputs = model(inputs)
#         outputs = outputs.to(device, torch.float)
#
#         loss = loss_func(outputs, labels)
#         loss.backward()
#
#         optimizer.step()
#         running_loss += loss.item()
#         if index % 1000 == 999:
#             last_loss = running_loss / 1000
#             print(f'  Batch #{index + 1} loss: {last_loss}')
#             tb_x = epoch_index * len(train_loader) + index + 1
#             tb_writer.add_scalar('Loss/train', last_loss, tb_x)
#             running_loss = 0.0
#     return last_loss
#
#
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
# EPOCH = 5
#
# best_val_loss = 1_000_000.
# for current_epoch in range(EPOCH):
#     print(f'EPOCH {current_epoch + 1}')
#
#     model.train(True)
#     avg_loss = epoch(current_epoch, writer)
#     model.train(False)
#
#     running_val_loss = 0.0
#     for val_index, val_data in enumerate(val_loader):
#         val_inputs, val_labels = val_data
#         val_inputs = val_inputs.to(device, torch.float)
#         val_labels = val_labels.to(device, torch.float)
#
#         val_outputs = model(val_inputs)
#         val_outputs = val_outputs.to(device, torch.float)
#
#         val_loss = loss_func(val_outputs, val_labels)
#         running_val_loss += val_loss
#
#     avg_val_loss = running_val_loss / (val_index + 1)
#     print(f'LOSS train {avg_loss} valid {avg_val_loss}')
#
#     # Log the running loss averaged per batch
#     # for both training and validation
#     writer.add_scalars('Training vs. Validation Loss',
#                        {'Training': avg_loss, 'Validation': avg_val_loss},
#                        current_epoch + 1)
#     writer.flush()
#
#     if avg_val_loss < best_val_loss:
#         best_vloss = avg_val_loss
#         model_path = 'model_{}_{}'.format(timestamp, current_epoch)
#         torch.save(model.state_dict(), model_path)