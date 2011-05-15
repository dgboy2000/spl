function compare_files(){
  FILE=$1 #'cccp.model'
  EXPEXTED_FILE=$2 #'cccp.model.expected'
  model_diff=`diff $FILE $EXPEXTED_FILE`
  if [ ${#model_diff} -gt 0 ]
  then
    echo "FAILED: different files $FILE and $EXPEXTED_FILE: $model_diff"
  else
    echo "PASS: files equal: $FILE and $EXPEXTED_FILE"
  fi
  
}

export -f compare_files