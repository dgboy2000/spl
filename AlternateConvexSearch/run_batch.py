# -script submits a batch of learning jobs with specified params, and data sets, folds, random seeds. script puts everything from the same run in a single directory, with a file containing the run parameters, and description of the run
# -then, run inference jobs; farm out an inference job to the same machine as a learning job. maybe use screen to persist the sessions? run inference on both the training and test datasets.
# -third phase: gather all the result files (test error/train error will be in files with just one number, the error rate; objective value will be in .time files: objective, time for run)

import datetime
import math
import os
import sys

print sys.argv
assert len(sys.argv) == 2, "Correct usage is python run_batch.py [params file]"
execfile(sys.argv[1])

make_cmd = "make clean && ./run_make"
status = os.system(make_cmd)
if status:
  print "MAKE COMMAND FAILED: "+make_cmd
  sys.exit(1)

# generate a directory that will hold all output for this run
extension = params['extension']
if extension == 'datetime':
  extension = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
  
dir_name = params['name']+'_'+extension
RUN_ROOT = 'runs/output/%s' %dir_name

if os.path.exists(RUN_ROOT):
  raw_input("WARNING: run directory %s already exists!  Press enter to continue working in this directory or Ctrl-C to quit" %RUN_ROOT)
else:
  os.mkdir(RUN_ROOT)

DATA_ROOT = params['data_path']
assert os.path.exists(DATA_ROOT)

# copy a README type file into the run directory
readme_filename = RUN_ROOT + '/README'
readme_explanation = "Below is a description of the run that created all the files in this directory:\n"
README_FILE=open(readme_filename, 'w')
README_FILE.write(readme_explanation)
README_FILE.write(params['description'] + "\n\n")
README_FILE.write("Here are the params used in the run:\n\n%s" %str(params))
README_FILE.close()


# process the params into a collection of the commands that will run everything
jobs = {}
for alg_name,param_pair in params['param_pairs'].iteritems():
  jobs[alg_name] = {}
  alg_jobs = jobs[alg_name]
  
  ALG_ROOT = RUN_ROOT+'/'+alg_name
  if not os.path.exists(ALG_ROOT):
    os.mkdir(ALG_ROOT)
  
  for prot in params['proteins']:
    alg_jobs[prot] = {}
    prot_jobs = alg_jobs[prot]
    
    for fold in params['folds']:
      prot_jobs[fold] = {}
      seed_jobs = prot_jobs[fold]
      
      for seed in params['seeds']:
        
        training_params, inference_params = param_pair
        training_params = training_params + ' --s '+seed
        
        training_data = '%s/train%s_%d.data' %(DATA_ROOT, prot, fold)
        test_data = '%s/test%s_%d.data' %(DATA_ROOT, prot, fold)        
        
        training_basename = '%s/motif%s_%d_%s' %(ALG_ROOT, prot, fold, seed)
        training_model = '%s.model' %training_basename
        training_job = "./svm_motif_learn %s %s %s %s"\
          %(training_params, training_data, training_model, training_basename)

        training_error_file = '%s.error.train' %training_basename
        training_inference_job = "./svm_motif_classify %s %s %s"\
          %(training_data, training_model, training_error_file)
        
        test_error_file = '%s.error.test' %training_basename
        test_inference_job = "./svm_motif_classify %s %s %s"\
          %(test_data, training_model, test_error_file)

        seed_jobs[seed] = [training_job, training_inference_job, test_inference_job]


def mean(a):
  return sum(a) / float(len(a))
  
def var(a):
  n = float(len(a))
  a2 = [x*x for x in a]
  return sum(a2) / n - sum(a)**2 / n**2

# Run the jobs
stats = {}
summary_stats = []
for alg_name, alg_jobs in jobs.iteritems():
  ALG_ROOT = RUN_ROOT+'/'+alg_name
  stats[alg_name] = {}
  alg_stats = stats[alg_name]
  
  for prot, prot_jobs in alg_jobs.iteritems():
    prot_stats = {'train' : [], 'test' : []}
    for fold, fold_jobs in prot_jobs.iteritems():
      for seed, seed_jobs in fold_jobs.iteritems():
        
        try:
          job_cmd = ' && '.join(seed_jobs)
        except TypeError:
          import pdb; pdb.set_trace()
        print "Executing the following command: %s\n" %job_cmd
        
        status = os.system(job_cmd)
        if status:
          if 'raise_errors' in params and params['raise_errors']:
            raise RuntimeError("COMMAND FAILED: "+job_cmd)
          else:
            print "COMMAND FAILED: "+job_cmd
            continue
          
        training_basename = '%s/motif%s_%d_%s' %(ALG_ROOT, prot, fold, seed)
        training_error_file = '%s.error.train' %training_basename
        test_error_file = '%s.error.test' %training_basename
        prot_stats['train'].append(float(open(training_error_file, 'r').read()))
        prot_stats['test'].append(float(open(test_error_file, 'r').read()))
        
    summary_stats.append("Stats for protein %s:" %prot)
    summary_stats.append("Training: n %d, mean %f, stdev %f" %(len(prot_stats['train']), mean(prot_stats['train']), math.sqrt(var(prot_stats['train']))))
    summary_stats.append("Test: n %d, mean %f, stdev %f" %(len(prot_stats['test']), mean(prot_stats['test']), math.sqrt(var(prot_stats['test']))))
    alg_stats[prot] = prot_stats

print "\n".join(summary_stats)
stats_filename = RUN_ROOT+'/STATS'
STATS_FILE = open(stats_filename, 'w')
STATS_FILE.write("\n".join(summary_stats))
STATS_FILE.write(str(stats))
STATS_FILE.close()
        

# def is_error_filename(filename):
#   if len(filename) < 12:
#     return false
#   if filename[-12:] == '.error.train' or filename[-11:] == '.error.test':
#     return true
#   return false
# 
# # Summarize the results
# for filename in os.listdir(RUN_ROOT):
#   alg_dir = RUN_ROOT+'/'+filename
#   if os.path.isdir(alg_dir):
#     alg_files = os.listdir(alg_dir)
#     for alg_file in alg_files:
#       if not is_error_filename(alg_file):
#         continue
#       
#       error = open(alg_file, 'r').read()
#       
  
  
  
  
  
  
  
  
  
  

