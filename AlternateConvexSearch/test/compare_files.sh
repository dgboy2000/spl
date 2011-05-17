function compare_files(){
  FILE=$1 #'cccp.model'
  EXPECTED_FILE=$2 #'cccp.model.expected'
  model_diff=`diff $FILE $EXPECTED_FILE`
  
  if [ ! -f $FILE ]
  then
   echo "FAILED: file $FILE doesn't exist"
   return
  fi
  
  if [ ! -f $EXPECTED_FILE ]
  then
   echo "FAILED: file $EXPECTED_FILE doesn't exist"
   return
  fi
  
  if [ ${#model_diff} -gt 0 ]
  then
    echo "FAILED: different $FILE and $EXPECTED_FILE: $model_diff"
  else
    echo "PASS: files equal: $FILE and $EXPECTED_FILE"
  fi
  
}

export -f compare_files