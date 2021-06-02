import tensorflow as tf
import os
import agent

# used in case we make changes to the graph of agent.py that don't require changing the actual variables,
# and we want to load a past run (i.e. add debugging checks)


agent_model = tf.saved_model.load(agent.model_savepath)

fn  = 'PongNoFrameskip-v42021-05-29 15:16:24.692423'

dir_path = os.path.dirname(os.path.realpath(__file__))
load_savepath = os.path.join(dir_path, 'saves/' + fn)
load_model = tf.saved_model.load(load_savepath)

for i,v in enumerate(load_model.vars):
  agent_model.vars[i].assign(v)

#agent_model.opt.set_weights(load_model.opt.get_weights())
#agent_model.lr = load_model.lr

tf.saved_model.save(agent_model, agent.model_savepath)

