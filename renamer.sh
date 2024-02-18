for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

echo "source_root = $source_root"
echo "destination_root = $destination_root"

# ----------------------------------------------------------------

mkdir -p $destination_root

for vid in $(find $source_root -type f -regex ".*/out_flow_\(test\|validation\)_[0-9]+.*" | sort); do

  basename $vid
    
  [[ $(basename $vid) =~ .*(test|validation)_([0-9]+)(.+) ]]
  split=${BASH_REMATCH[1]} 
  number=${BASH_REMATCH[2]} 
  extension=${BASH_REMATCH[3]}

  out_path="$destination_root/video_${split}_${number}${extension}"

  cp $vid $out_path

done
