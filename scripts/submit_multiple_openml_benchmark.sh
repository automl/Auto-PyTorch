#!/bin/bash

RUN=0
CONFIG=0
INSTANCE=0
export AUTONET_HOME=$PWD

mkdir outputs
cd outputs
while [ $INSTANCE -le 72 ]
do
  export INSTANCE
  export CONFIG
  export RUN
  OPDIR=output_${INSTANCE}_${CONFIG}_${RUN}
  mkdir $OPDIR
  cp $AUTONET_HOME/scripts/run_openml_benchmark.moab $OPDIR
  cd $OPDIR
  export OUTPUTDIR=$PWD
  msub run_openml_benchmark.moab | sed '/^\s*$/d' >> $AUTONET_HOME/jobs.txt
  cd ..
  let INSTANCE=$INSTANCE+1
done
