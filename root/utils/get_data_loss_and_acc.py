import json

with open("loss.json", "r") as my_file:
    loss_json = my_file.read()
with open("acc.json", "r") as my_file:
    acc_json = my_file.read()
with open("avg_loss.json", "r") as my_file:
    avg_loss_json = my_file.read()
with open("avg_acc.json", "r") as my_file:
    avg_acc_json = my_file.read()

loss_json = json.loads(loss_json)
acc_json = json.loads(acc_json)
avg_loss_json = json.loads(avg_loss_json)
avg_acc_json = json.loads(avg_acc_json)

# def read_loss_and_acc():
#     with open("loss.json", "r") as my_file:
#         loss_json = my_file.read()
#     with open("acc.json", "r") as my_file:
#         acc_json = my_file.read()
#     with open("avg_loss.json", "r") as my_file:
#         avg_loss_json = my_file.read()
#     with open("avg_acc.json", "r") as my_file:
#         avg_acc_json = my_file.read()
#
#     loss_json = json.loads(loss_json)
#     acc_json = json.loads(acc_json)
#     avg_loss_json = json.loads(avg_loss_json)
#     avg_acc_json = json.loads(avg_acc_json)
