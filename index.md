## Coming to Climate Conclusions

Over thanksgiving break, I found myself having to repeatedly explain why climate change is an issue, and why its definitely caused by humans this time. While I think I made some convincing arguments, I really wished I had more good, data-proven visualizations to help get my point across. Having a transparent, reproducible graph is more convincing than just, "because the scientists say so". Thats why in preperation for winter break, I made a quick sample on how to handle arguments with a data-science approach. I hope this walthrough can give readers ammunition in any climate change arguments that may arise this holiday season, but more importantly, I want readers to learn how to take raw data and draw conclusions themselves.

Lets take this statement for example: 
"Earths climate has been changing throughout all of history, there are natural cycles that are thousands of years long. The warming we see now could be caused by things like earths orbit changing" 

Quick googling nets me the following information: 
We detect largescale cycles by looking at rock layers. The climactic cycles caused by changes in earths orbit are called Milankovitch cycles. There are three components of Milankovitch cycles: 

- eccentricity: cycles about 100,000 years
- obliquity: cycles about 41,000 years
- precession: cycles about 25,771.5 years

Eccentricity describes the shape of our orbit; if its more circular or more elliptical. Obliquity describes the tilt of our planets axis. Precession is the "wobble" of our planet, simlilar to how a spinning dreidle sometimes wobbles. Here is an article that fully explains milankovitch cycles and even has videos showing what each type of cycle looks like: https://climate.nasa.gov/news/2948/milankovitch-orbital-cycles-and-their-role-in-earths-climate/ 

So, can these cycles really explain climate change? Lets visualize the data to find out more about the bahavior of milankovitch cycles. 

```python3
import pandas as pd 
# ^ this library is for dealing with dataframes, check it out: https://www.w3schools.com/python/pandas/default.asp 

# data from: http://vo.imcce.fr/insola/earth/online/earth/La2004/INSOLN.LA2004.BTL.ASC 
# or more generally, from this research project: http://vo.imcce.fr/insola/earth/online/earth/earth.html 
df = pd.read_fwf('milankov_data.txt')

display(df)
```
![image](https://user-images.githubusercontent.com/49928811/146706472-86dc875f-c32d-4a6b-a87b-860df481a996.png)

```python3

# dont need THAT much past data, I think 5 million years will suffice?
df = df[df['time'] >= -5000]


# we have to convert the data to numerical data so that we can use it
# notice that the format of the numbers is weird; ends with something like "D-01"
# the D means the number is a double, and the number after that is how many places 
# we need to move the decimal

import re 
# ^ regular expressions, just a way of parsing text: https://www.computerhope.com/jargon/r/regex.htm   

# this function converts the data to values we can use
def to_numeric(input_str):
    result = re.search('0[.](\d+)D([-+])\d(\d)', input_str)
    num_str = result.group(1)
    plus_or_minus = result.group(2)
    move_decimal = int(result.group(3))
    # sanity check
    # print(f"num string is {num_str}, decimals is {move_decimal}")
    if plus_or_minus == '-':
        while move_decimal > 0:
            num_str = '0'+ num_str
            move_decimal -= 1
        num_str = '.' + num_str
    elif plus_or_minus == '+':
        num_str = num_str[0:move_decimal] + '.' + num_str[move_decimal:]
    return float(num_str)
    
df['ecc'] = df['ecc'].apply(lambda x: to_numeric(x))
df['obl'] = df['obl'].apply(lambda x: to_numeric(x))
df['pre'] = df['pre'].apply(lambda x: to_numeric(x))

print(df.head()) #looks good!
```
![image](https://user-images.githubusercontent.com/49928811/146706539-94da5484-4277-46f8-a708-8ad93d02a26d.png)

```python3
# plotting the different milankovitch forcings 

df.plot(x='time', y='ecc', legend=False).set_title("Earths eccentricity over the last 5 million years")

#cut down on time again, to 1 million years 
df = df[df['time'] >= -1000]
df.plot(x='time', y='obl', legend=False).set_title("Earths obliquity over the last 1 million years")

#cut down on time again, to 500 thousand years 
df = df[df['time'] >= -500]
df.plot(x='time', y='pre', legend=False).set_title("Earths precession over the last 500,000 years")

#what does the combination of all three look like? 
import matplotlib.pyplot as plt # need a different plotting tool, its more general

#smallest possible timeframe we can look at 
df = df[df['time'] >= -1]
plt.show() #clears the plot
combined = [row['ecc'] + row['obl'] + row['pre'] for _, row in df.iterrows()]
plt.plot(df['time'], combined)
plt.title("Milankovitch Forcing over the last 1,000 years")
```
![image](https://user-images.githubusercontent.com/49928811/146706602-c06078a8-2a0e-4272-a273-a7696ded5cd5.png)![image](https://user-images.githubusercontent.com/49928811/146706633-3978aa86-59e4-4c35-8f38-c3e967030d0c.png)![image](https://user-images.githubusercontent.com/49928811/146706654-318774a1-f6ac-444c-8795-9a6318338f6d.png)![image](https://user-images.githubusercontent.com/49928811/146706691-40673920-376a-4409-b611-219930c069bc.png)

From these graphs, we can get an idea of the timescales and general behavior of milankovitch cycles. Note that the data we have for the milankovitch cycles is per every thousand years (and the values dont change much between any given thousand years), while this climate change is happening in a matter of decades. That last graph seems to support the argument that our climate could be changing due to milankovitch cycles, right? Sometimes its important to plot things into perspective:

```python3
import numpy as np

# get slope of the total milankovitch forcing from the last graph 
slope = (combined[0] - combined[1])/1000
# ^ represents the current change in milankovitch forcing per year
print(f"slope is: {slope}\n")

years = np.linspace(1760, 2020, 260)
forcings = [combined[1]]
for i in range(len(years)-1):
    forcings.append(forcings[i] + slope)

plt.plot(years, forcings)
#make sure to put the change in forcing into perspective;
#y values are relative to the total fluctuation
plt.yticks(np.arange(0,7))
plt.title("Milankovitch Forcing Since the Industrial Revolution")
plt.xlabel('year')
plt.show()

# lets plot global temperature too for comparison
temp_df = pd.read_fwf('nasaGlobalTemp.txt')
temp_df.plot(x='Year', y='No_Smoothing', legend=False, title="Average Global Temperature", xlabel="year")
```
![image](https://user-images.githubusercontent.com/49928811/146706755-caceffa2-a65f-452f-b8ad-59a72f15a388.png)![image](https://user-images.githubusercontent.com/49928811/146706768-5bc55f0c-7800-4687-ab19-b208d1b16a4b.png)

Given that the forcing from milankovitch cycles is nearly constant while the global temperature looks exponential, cycles that take tens of thousands of years can safely be ruled out. However, there are also decadal cycles on earth that are caused by long-term patterns of high and low pressure zones. A significant one that comes to mind is the pacific decedal oscillation. Watch this two minute video to get a better understanding of how it operates: https://www.youtube.com/watch?v=Sc3tOEcM0YE 
Next up, we'll be using some pretty neat statistics to give us a better understanding of the correlation between the pdo and the global temperature.

```python3
# data from https://www.ncdc.noaa.gov/teleconnections/pdo/ 
pdo_df = pd.read_fwf('noaaPDOdata.txt')
display(pdo_df)
```
![image](https://user-images.githubusercontent.com/49928811/146706891-b94c68a3-72c1-4575-a7e3-bccd809ed5cb.png)
```python3
# get the yearly average 
pdo_df['average'] = pdo_df[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']].mean(axis=1)


# theres a small issue with the data: Dec of 2021's PDO is set to 99.99, probably because december of 2021 isnt over yet.
# lets just exclude 2021 from our analysis since we dont yet have its complete data.
pdo_df.drop(pdo_df.tail(1).index,inplace=True)

pdo_df.plot(x='Year', y='average', label='pdo', title="Pacific Decedal Oscillation Forcing")

# since the pdo forcing is so chaotic, lets approximate it
# smoothing it out will make it much easier to look at when we look 
# at pdo and temperature side by side 
approx = np.poly1d(np.polyfit(pdo_df['Year'], pdo_df['average'], 8))
plt.plot(pdo_df['Year'], approx(pdo_df['Year']), label='approximation of pdo')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/49928811/146707053-8143f4b4-2ca2-4cd8-a416-84a105fa76c4.png)
```python3
#drop data from before 1880 since thats the earliest global temp data we have\
pdo_df = pdo_df[pdo_df['Year'] >= 1880]

plt.plot(temp_df['Year'], temp_df['Lowess(5)'], label="global temperature")
plt.plot(pdo_df['Year'], approx(pdo_df['Year']), label="PDO approximation")
plt.legend()
plt.title("PDO forcing and global temperature")
```
![image](https://user-images.githubusercontent.com/49928811/146707103-e65d3fe6-7f75-423f-9971-9e6af48d2f60.png)



