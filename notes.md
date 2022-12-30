# Particular notes of the study

## Univariate study

1. Most cases has positive values 
2. submetering in columns 7,8,9 has a skew distribution, with really long tails for every case, particular the submetering in 9 has a range between 16 and 20 with more cases, can be considered with a bimodal distribution.
3.  Global intensity (6) follow similar rule
4. Voltage for some reason has a lot outliers [-20,0) around 26k
5. Glabal active and reactive power have similar distribution, but second one have several outliers

## bivariate study

1. Correletion heatmap show some strong correlation between:
	a. Global activity and global intensity
	b. Global reactive power and global Voltage
	c. Global activity and sub metering 3
but is a) who shows the more linear behaviour trend expected.

## Reviewing ts series
1. Year 2006 is more noisy for first variable
2. searching for average the voltage show clear trends for months and in july show a clear down trend, probably is the hot season (summer)

# Missing values
1. The missing data is represented by -1 in data frame by all row
2. Every year is increasing the amount of data missing
3. there is no big difference in hour
4. high presence of missing values on weekend if compare with the rest of week.

*This can lead to a reason where there is more presence of people in the house but is just a theory for now*


