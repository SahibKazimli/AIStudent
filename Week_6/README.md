# Week 6
We discussed on how to run processing pipelines on Modal. Here you'll try to analyse the systems by tweaking some parameters and understanding the results.

## Task 1
Try to use Modal's metrics and analyse the CPU, RAM and GPU utilisation for the current process. Is that optimal? Try to find modal documentation on how to limit CPU and RAM for a given app. What if you don't use a GPU?

## Task 2
We had a very simple schedule to run the app. What if there is a complex schedule like every other day of the week(Mon, Wed, Fri)? Explore Cron and try to schedule tasks for complex schedules.

## Task 3
We have used a `batch_size` parameter and set it $8$. Why $8$? Try bigger batch sizes and see how the system usage changes. Compare execution times, RAM usage and GPU VRAM usage for atleast 5 different batch sizes?

## Task 4
Repeat the above Tasks 1 and 3 with the 100k dataset and find out why it fails. **Hint** - Data gets 10 times larger but the GPU memory stays the same. What if you don't use GPU/cuda?

## Resources 

CRON - https://crontab.guru/

MODAL Docs - https://modal.com/docs

Datasets - https://drive.google.com/drive/folders/1_rvfHGVRe-fI1_RQAujFFPhAZ2AlbQKK?usp=sharing