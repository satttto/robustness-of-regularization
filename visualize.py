
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm

with open("attack_result.json","w") as f:
  data = json.load(f)

print(data)

# visualize
plt.figure()
plt.title(title)
for i in range(len(model_param_paths)):
    label = model_param_paths[i].rsplit('/', 1)[1]
    plt.plot([x * 0.01 for x in range(0, 6)], results[i][''], color=cm.jet(1-i/10.0), label=label)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Adversarial Example Epsilon')
plt.ylabel('Accuracy')
plt.grid()
plt.show()