# Data

Ok so the spikes are extracted from an LFP recording. So in the LFP recording, you will also get the spikes. If I remember correctly you can find the times of spike initiation in the file spike_times and the clusters in spike_clusters (so which spike belongs to which unit).
My previous student had some really pretty code to deal with all of this already but at the moment I don't have time to get back into it, please forgive me..

The LFP data is in continuous.dat which is in binary. In the package I developed, I have a class to deal with the importing of this file.
Here is a mock script to show you how to import them. We only have one channel that was recording. https://we.tl/t-axMobZrzH9

Everything is based on neurokin, I suggest you clone the package: https://github.com/WengerLab/neurokin in theory the main branch should work, but the most up-to-date is feature_call_from_kindata (I know it's terrible, I need to get my code properly in shape).
You'll probably have to install 2 packages: tdt and fooof (I guess you have this one already)

I hope everything works out!

Best,
Elisa