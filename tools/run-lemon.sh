SOLVER=/home/pmazumder/lemon-1.3.1/build/tools/dimacs-solver
OUTDIR=../lemon-runs

set -x

mkdir -p $OUTDIR

for MINGZ in $(ls -S -r /tmp/data/*.min.gz); do
  MIN=${MINGZ%.*}
  OUTF=$OUTDIR/$(basename $MIN).run
  if [ -f $OUTF ]; then
    if grep -q "Min flow cost" $OUTF; then
      continue
    fi
  fi
  echo "processing $MINGZ..."
  gunzip -k -f $MINGZ > $MIN
  $SOLVER $MIN > $OUTF 2>&1
  rm $MIN
done
exit

