mkdir females males idk



for i in `cat fnames`; do
	mv -v lfwcrop_grey/faces/$i\_* ./females/
done


for i in `cat mnames`; do
	mv -v lfwcrop_grey/faces/$i\_* ./males/
done


for i in `cat nnames`; do
	mv -v lfwcrop_grey/faces/$i\_* ./idk/
done

mv -v lfwcrop_grey/faces/* ./idk/
