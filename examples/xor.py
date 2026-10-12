"""Train XOR through the same graph and reverse-mode engine used in the browser."""
from forge_ml.model import Experiment

experiment=Experiment({'dataset':'xor','hidden':[16,16],'seed':42,'noise':.08})
for _ in range(10):
    experiment.cpu_step(30)
    ids=experiment.data['splits']['validation']
    result=experiment.evaluate(experiment.data['x'][ids],experiment.data['y'][ids])
    accuracy=(result['probabilities'].argmax(1)==experiment.data['y'][ids]).mean()
    print(f"step={experiment.step:3d} loss={float(result['loss']):.6f} validation accuracy={accuracy:.3f}")
