# From Data to Justice

<p align="center">
  <a href="https://go-skill-icons.vercel.app/">
    <img src="https://go-skill-icons.vercel.app/api/icons?i=r,py" />
  </a>
</p>

This Project alings with the Goal of [Murder Accountability Project](https://www.murderdata.org/), in that we have developed Machine Learning models that can detect Offender Sex and Offender Age from already solved cases.

![crime](/images/crime_rate.png)

---

## Dataset

[Data](https://www.murderdata.org/p/data-docs.html) is again taken from Murder accountability projects and contains details of over **870k** homicide cases reported from 1976 to 2022, including solved and unsolved cases with 30 features.

Features includes Victim Sex, Victim Age, Year, Month, Agency, Weapon used, Victim Race, Victim Ethnicity, State, City, and so on.

---

### EDA

Exploratory data analysis was conducted with help of R present in **EDA_R_code.qmd** to determine trends in data and seeing importance of various features for prediction.

---

#### 1. Solved and unsolved cases:

![gauge](/images/solved_unsolved.png)

We can see that only **70.5%** cases are solved that means their are over **256k** cases are still unsolved.

---

#### 2. Sate with most homicide cases:

![Homi_states](/images/Homi_states.png)

Graphs shows that **California** and **Texas** are most notable out of all 50 US states.

---

#### 3. Sates unsolved homicides:

![unsolved_homi](/images/unsolved_states.png)

Again **Claifornia** and **Texas** are notable but also **New York** state.

---

#### 4. Weapons of choice:

![wordcloud](/images/wordcloud.png)

Most popular weapon of choice is **Handgun**.

---

#### 5. Relations between Offender and Victim:

![relations](/images/relations_bar.png)

- Relations are combined for better viewing. It is observed that solved cases have relationship distributed while in unsolved cases it's simply **Unknown**.
- This shows that **Relationship** is not an important feature in unsolved cases.

---

#### 6. Victim and Offender Ages:

![Age_dist](/images/pop_pyr.png)

- We can see high number of victims and Offenders between 20 to 40, and high number in NA column is contributed by Offenders unknown age in Unsolved cases.
- Offender Age is one of the variables we predict and here we see some similar increase and decrease in it with Victim Age.
  
---

#### 7. Victim and Offender Gender:

![sex](/images/sex_pie.png)

- Most crimes are commited upon and by Males.
- Offender Gender is something we have predicted as we see large **Unknown** in Offender Sex distribution.

---

#### 8. Years:

![victimyear](/images/year1.png)

![offyear](/images/year2.png)

- We don't see any particular pattern in Victim Age and Offender Age over the year.
  
##### Maybe diving years into decades isn't effective lets try seasons with complex grapgs

![waffle](/images/waffle.png)

- Even this graph shows no particular pattern.
- So, **Year** is also eliminated from features list.

---

#### 9. Agency type:

![radar](/images/radar.png)

We can clearly see **Municipal police** has most number of homicide cases.

![radar2](/images/unsolved_radar.png)

- Graph unsurpisingly shows that **Municipal police** has highest **31.8%** of pending murder cases.
- Surprisingly **County police** and **Special police** dispite having lesser numbers of cases still have **31.1%** and **26%** of unsolved cases respectively.

---

#### 10. States and Offenders:

![states](/images/offstates.png)

We can see again that **Claifornia**, **Texas**, and **New York** have high number of Unkown gender which means State is important variable.

---

#### 11. Race:

- **Offender Race and Ethnicity** are known in unsolved cases for obvious reasons.
- We can still invastigate **Victim Race** in such as well as solved cases.
  
![unolved_dot](/images/unsolved_dot.png)

![solved_dot](/images/solved_dot.png)

- We can clearly that pattern matches in unsolved and all cases, so we can't ignore Victim Race.

---

### Machine Learning

- Here we levrage power of Python to train model as displayed in **notebook.ipynb** to create algorithms to predict Offender Age and Offender Sex.
- Due to **complex nature** of data and **categorical columns having small unique patterns** with some with high cardinality using linear methods was discarded.
- Tree based methods like decision trees, XGBoost and Random forests are more viable.

---

#### Feature Selection

- After no longer considering linear methods we need to next select features that can be used for training out **tree based regressor to predict offender age** and **tree based calssifier to predict their gender.**

- We use permutation feature importances to eliminate effect of high cardinality of features after removing these variables as a result of EDA:
    - ID: unique IDs of cases.
    - Solved: all rows are "Yes".
    - VicEthnic, OffEthnic: Most rows are "Unknown or not reported".
    - OffRace, VicCount, OffCount: Unsolved cases have empty rows.
    - Relationship, Subcircum, Circumstance, Situation: Unsolved cases have it set as "Unknown" or equivalent.
    - FileDate: Just Date.
    - MSA: CNTYFIPS + State.
    - Ori: Source + Agency.
    - Incident: Number of case within a month.
    - CNTYFIPS, Agency: Very high cardinality.

- Decision Tree Regressor(Offender Age) Permutaion importances:
    i.      VicAge: 0.3906
    ii.     Weapon: 0.1224
    iii.    Year: 0.0898
    iv.     VicSex: 0.0778
    v.      State: 0.0552
    vi.     VicRace: 0.0465
    vii.    OffSex: 0.0435
    viii.   Agentype: 0.0295
    ix.     Month: 0.0179
    x.      ActionType: 0.0120

- Decision Tree Classifier(Offender Sex) Permutation importances:
    i.      Weapon: 0.0144
    ii.     VicAge: 0.0139
    iii.    OffAge: 0.0116
    iV.     VicSex: 0.0098
    v.      Year: 0.0092
    vi.     State: 0.0060
    vii.    VicRace: 0.0056
    viii.   Agentype: 0.0023
    ix.     Month: 0.0019
    x.      Homicide: 0.0010

- We select **top 9 features** except OffAge and OffSex, for in training of both algorithms.
  
---

#### Initial Model results

-  Regression (Offender Age prediction):
   - XGBoost
     - RMSE: 11.24 years
     - R-squared: 25%
     - Adjusted R-squared: 25%
   - Random forest
     - RMSE: 11.45 years
     - R-squared: 22%
     - Adjusted R-squared: 22% 

- This shows that dataset has much more unqiue and smaller quantity of categories in columns that going untraced.
- So, we don't bother with Classification and move to next step.

---

### Model Results after Oversampling

- We use **SMOTE(Synthetic Minority Oversampling Technique)**, which is generally used for classification tasks in both models.
- Regression (Offender Age):
  - XGBoost:
    - R-squared: 79.79%
    - Adjusted R-squared: 79.78%
    - RMSE: 12.97 Years
  - Decision Tree:
    -  R-squared: 84.83%
    -  Adjusted R-squared: 84.37%
    -  RMSE: 11.41 years

- Classification (Offender Sex):
  - Random Forest:
    - Accuracy: 92.72%
    - Precision: 92.87%
    - Recall: 92.72%
    - F1 Score: 92.72%
  - XGBoost:
    - Accuracy: 91.96%
    - Precision: 92.34%
    - Recall: 91.69%
    - F1 Score: 91.66%

---
