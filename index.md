## Coming to Climate Conclusions
written by Veronika Polushina

Over thanksgiving break, I found myself having to repeatedly explain why climate change is an issue, and why its definitely caused by humans this time. While I think I made some convincing arguments, I really wished I had more good, data-proven visualizations to help get my point across. Having a transparent, reproducible results is a great way to convince others. Even just getting a better understanding of how scientists come to their conclusions could reduce skeptism of the results. Thats why in preperation for winter break, I made a quick sample on how to get real insight from messy data. I hope this walkthrough can give readers ammunition in any climate change arguments that may arise this holiday season, but more importantly, I want readers to learn how to take raw data and draw conclusions themselves.

Lets take this statement for example: 

"Earths climate has been changing throughout all of history, there are natural cycles that are thousands of years long. The warming we see now could be caused by things like earths orbit changing" 

Quick googling nets me the following information: 

We detect largescale cycles by looking at rock layers. The climactic cycles caused by changes in earths orbit are called Milankovitch cycles. There are three components of Milankovitch cycles: 

- eccentricity: cycles about 100,000 years
- obliquity: cycles about 41,000 years
- precession: cycles about 25,771.5 years

Eccentricity describes the shape of our orbit; if its more circular or more elliptical. Obliquity describes the tilt of our planets axis. Precession is the "wobble" of our planet, simlilar to how a spinning dreidle sometimes wobbles. [Here](https://climate.nasa.gov/news/2948/milankovitch-orbital-cycles-and-their-role-in-earths-climate/) is an article that fully explains milankovitch cycles and even has videos showing what each type of cycle looks like.  

So, can these cycles really explain climate change? Lets visualize the data to find out more about the bahavior of milankovitch cycles. 

[milankovitch cycle data](http://vo.imcce.fr/insola/earth/online/earth/La2004/INSOLN.LA2004.BTL.ASC)

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

# notice that there arent any None / null values in this dataset;
# dont forget to check datasets (or use the .notna() function)

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

Given that the forcing from milankovitch cycles is nearly constant while the global temperature looks exponential, cycles that take tens of thousands of years can safely be ruled out. However, there are also decadal cycles on earth that are caused by long-term patterns of high and low pressure zones. A significant one that comes to mind is the pacific decedal oscillation. I highly reccomend watching [this two minute video](https://www.youtube.com/watch?v=Sc3tOEcM0YE) to get a better understanding of how the pacific decedal oscillation (PDO) operates. 
Next up, we'll be using some pretty neat statistics to give us a better understanding of the correlation between the PDO index and the global temperature.

[PDO data](//www.ncdc.noaa.gov/teleconnections/pdo/)
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

They dont look very correlated, but thats not enough to draw conclusions. There are many statistical analyses that can give us more insight, but lets start with some basic ones.
The pearson correlation coefficient can tell us how strongly two continuous variables are correlated. It ranges from -1 to 1, -1 being a strong inverse correlation and 1 being a strong positive relationship. A pearson correlation around zero indicates little to no correlation. 

p-values are a little harder to explain, [this article](https://www.scribbr.com/statistics/p-value/) probably does it better than I can. 
tldr: the p value is the probability the observation happened by chance. A smaller p-value suggests higher statistical significance 

```python3
covariance = np.cov(temp_df['No_Smoothing'], pdo_df['average'])
print(covariance) #matrix

from scipy.stats import pearsonr

pears_corr_coef, p_value = pearsonr(temp_df['No_Smoothing'], pdo_df['average'])
print(f"Pearsons correlation: {pears_corr_coef}\nP-value: {p_value}\n")
```
![image](https://user-images.githubusercontent.com/49928811/146712431-6e9d8813-0d10-4e4b-a331-8b82578774c3.png)

The pearsons correlation is very small, which suggests that the pacific decedal oscillation is not significantly contributing to the rapid warming of our planet. However, our p-value says that there is 73% likelihood that the observation is by chance. Typically, the null hypothesis is only rejected if there is a less than 5% chance (so a p-value < .05) Here, we do not yet have enough data to come to a conclusion. 

If we cant tell whether the pacific decedal oscillation is causing an increase in global temperature over the last couple of decades, maybe we can say with greater confidence that carbon dioxide emmissions are? 

[co2 data](https://github.com/owid/co2-data/blob/master/owid-co2-data.csv)
```python3
#co2 emmisions 
#data from: https://github.com/owid/co2-data/blob/master/owid-co2-data.csv 
co2_df = pd.read_csv('co2info.csv')

# display(co2_df)
co2 = np.array(co2_df.groupby(['year'])['co2'].mean())

#years 1750- 2020
years = list(range(1750, 2021))

plt.plot(years, co2, label="human co2 emission rate")
plt.ylabel("parts per million")
plt.xlabel("year")
plt.title("Human CO2 Emissions")
#note the difference between emissions and atmosphere level
#note the difference between total and per year 
plt.show()
```
![image](https://user-images.githubusercontent.com/49928811/146712757-f089b14e-1bad-41f8-9cad-eb4b80b03141.png)
 
 This curve does seem to resemble the gloabl temperature trend better. What do the numbers have to say? 
 
 ```python3
 # human emissions vs temperature correlation 

#since we only have temperature data starting in 1880, gotta cut the co2 data down
co2 = co2[130:]
years = years[130:]

pears_corr_coef, p_value = pearsonr(list(temp_df['No_Smoothing']), co2)
print(f"Pearsons correlation: {pears_corr_coef}\nP-value: {p_value}\n")
 ```
![image](https://user-images.githubusercontent.com/49928811/146713899-09ed1821-920d-4b05-8d8d-b76d4056dca0.png)

Wow! I've never seen a p value that low. Its in scientific notation, so there are 53 zeros between the decimal and that first "2". The pearsons correlation is close to 1, so we can be very certain that an increase in carbon emissions is correlated with an increase in global temperature. 

The next step is to try to use this data to predict various outcomes. There are many, many different types of machine learning algorithms we can use. Generally you try simple methods, and if they arent enough you use more complex things to suit your needs. We'll begin with one of the simple machine learning models, linear regression, to try and predict global temperatures using human carbon dioxide emissions. A linear regression just tries to find a line that best fits the data points by minimizing the vertical distance between the data points and the regression line. Heres a neat visual of the least sqaures method:

![image](https://user-images.githubusercontent.com/49928811/146828938-09805b77-ce1a-47f6-adf3-66be2301718f.png)
[Source](https://medium.com/analytics-vidhya/ordinary-least-square-ols-method-for-linear-regression-ef8ca10aadfc)

you can read more about linear regressions and their purpose in machine learning [here](https://machinelearningmastery.com/linear-regression-for-machine-learning/) 

```python3
from sklearn.linear_model import LinearRegression

#reshaping our data so that it follows the #samples by #features format
co2 = co2.reshape(-1, 1)
global_temps = np.array(temp_df['No_Smoothing']).reshape(-1, 1)

reg = LinearRegression().fit(co2, global_temps)
print(f"score is: {reg.score(co2 ,global_temps)}")
print(f"coefficient is: {reg.coef_[0]}")
print(f"intercept is: {reg.intercept_}")
```
![image](https://user-images.githubusercontent.com/49928811/146820254-8146776e-11ce-4295-8d16-9dd67e992862.png)

The score tells us that roughly 82% of the observed variation can be explained by the model's inputs (the best possible score is 1, which would indicate that there is no unexplained variation.) [Heres](https://www.investopedia.com/terms/r/r-squared.asp) more information about what the score represents. 

The coefficient is the slope of the regression line. Combined with the intercept, we can extract the function of the regression line as: y = coef\*x + interpect
```python3
#prediction for the next 100 years if we cap yearly emissions to 500ppm
co2_hypothetical = np.array([500]*100).reshape(-1, 1)
future = range(2020,2120)

plt.plot(future, reg.predict(co2_hypothetical))
plt.title("Global Temperature With Capped Emissions")
plt.xlabel("year")
plt.show()

# if we decrease rate of emmissions by decade steps 
co2_hypothetical = np.array([500]*10 + [480]*10 + [460]*10 +[440]*10 + [420]*10 + [400]*10  + [380]*10 + [360]*10 + [340]*10 + [320]*10).reshape(-1,1)
plt.plot(future, reg.predict(co2_hypothetical))
plt.title("Global Temperature With Decreasing Emissions")
plt.xlabel("year")
plt.show()
```
![image](https://user-images.githubusercontent.com/49928811/146821927-2c28b3f6-2cb7-4eea-ae06-38eaa0bb6466.png)
![image](https://user-images.githubusercontent.com/49928811/146821947-6f21ca2f-2ab5-43b2-859f-d6d69b2e5d7e.png)

Do you think this is what the global temperature trends should look like? Think about it- if we cap the amount of emmissions per year, the amount of co2 in the atmosphere would still be increasing, and therefore the temperature would also still be increasing. There are some serious flaws with using a linear regression in this context. We know there is a strong correlation between human co2 emissions and global temperatures, but that does not mean co2 emisions cause the global temperature to rise. A linear model assumes a linear relationship beween the variables, which is not true in this case. In reality, predicting the gloabl temperatures has a lot of factors, more than just emissions. For example, the average temperature even affects itself; warmer temperature -> less snow and ice coverage of our planet -> lower planetary albedo (albedo is the amount of radiation reflected instead of absorbed) -> warmer temperatures. We'd need a combination of different models, and probably a different machine learning model in general. I encourage readers to take a look at this python library just to know whats out there for you to use: [sklearn](https://scikit-learn.org/stable/) It has great reproducable examples for each tool.

At this point, I'd just trust the collective researchers that make accurate models of climate for a living. It's still really useful to see the process (and maybe even replicate it) for yourself. Its important to keep in mind that while data visualization can be extremely revealing, it can also mislead. This quick walthrough is just sratching the surface of what you could do with data; if you want more control over your percpective (and the perspective of whoever you like to argue with), I highly reccomend taking this [course](https://cmsc320.github.io/). Happy holidays!

