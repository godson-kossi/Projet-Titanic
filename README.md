# Projet-Titanic

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "0253bee0-db52-470e-9c2e-8282deeb205b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# imports...\n",
    "from sklearn import datasets\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "49f65b6c-6787-4e8c-a68a-08c0be995ddd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('titanic.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "445b1c7f-d9f1-474e-8fe3-484751842afa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "raw",
   "id": "096bd4d7-482d-41a0-86c4-5e66ed9fb3de",
   "metadata": {},
   "source": [
    "Le nombre de ligne"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "acd865f8-bd6a-447b-b87a-e3e74dbc9edb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "891"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape[0]"
   ]
  },
  {
   "cell_type": "raw",
   "id": "344e8487-f41b-4100-b439-1587d914059c",
   "metadata": {},
   "source": [
    "Le nombre de colonnes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "ce71c584-a4f0-4764-af18-216662d9d8eb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "12"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape[1]"
   ]
  },
  {
   "cell_type": "raw",
   "id": "50ca5ada-d2c6-4f78-8e21-f05c055ee1a6",
   "metadata": {},
   "source": [
    "Les colonnes de 'titanic.csv'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "7fb6473e-c920-4785-af56-6a58855b2bb6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "raw",
   "id": "55e88b3b-4595-46e6-9544-7df2302a5df9",
   "metadata": {},
   "source": [
    "Le type de donnee dans chaque colonne"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8e8cffee-81bf-481d-8571-8f08eb433376",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "00b8f626-b4af-43d3-a421-3455a83e7ffd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method NDFrame.head of      PassengerId  Survived  Pclass  \\\n",
       "0              1         0       3   \n",
       "1              2         1       1   \n",
       "2              3         1       3   \n",
       "3              4         1       1   \n",
       "4              5         0       3   \n",
       "..           ...       ...     ...   \n",
       "886          887         0       2   \n",
       "887          888         1       1   \n",
       "888          889         0       3   \n",
       "889          890         1       1   \n",
       "890          891         0       3   \n",
       "\n",
       "                                                  Name     Sex   Age  SibSp  \\\n",
       "0                              Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1    Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                               Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3         Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                             Allen, Mr. William Henry    male  35.0      0   \n",
       "..                                                 ...     ...   ...    ...   \n",
       "886                              Montvila, Rev. Juozas    male  27.0      0   \n",
       "887                       Graham, Miss. Margaret Edith  female  19.0      0   \n",
       "888           Johnston, Miss. Catherine Helen \"Carrie\"  female   NaN      1   \n",
       "889                              Behr, Mr. Karl Howell    male  26.0      0   \n",
       "890                                Dooley, Mr. Patrick    male  32.0      0   \n",
       "\n",
       "     Parch            Ticket     Fare Cabin Embarked  \n",
       "0        0         A/5 21171   7.2500   NaN        S  \n",
       "1        0          PC 17599  71.2833   C85        C  \n",
       "2        0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3        0            113803  53.1000  C123        S  \n",
       "4        0            373450   8.0500   NaN        S  \n",
       "..     ...               ...      ...   ...      ...  \n",
       "886      0            211536  13.0000   NaN        S  \n",
       "887      0            112053  30.0000   B42        S  \n",
       "888      2        W./C. 6607  23.4500   NaN        S  \n",
       "889      0            111369  30.0000  C148        C  \n",
       "890      0            370376   7.7500   NaN        Q  \n",
       "\n",
       "[891 rows x 12 columns]>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head"
   ]
  },
  {
   "cell_type": "raw",
   "id": "cc5b6ed6-2e0d-4b47-bbf7-a71221fb25d9",
   "metadata": {},
   "source": [
    "Le nombre des valleurs manquantes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9a708df1-67ee-4577-8d26-ac811f4fda5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      0\n",
       "Survived         0\n",
       "Pclass           0\n",
       "Name             0\n",
       "Sex              0\n",
       "Age            177\n",
       "SibSp            0\n",
       "Parch            0\n",
       "Ticket           0\n",
       "Fare             0\n",
       "Cabin          687\n",
       "Embarked         2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "raw",
   "id": "bd3cf382-9c60-45b6-9063-ca46159e9ff4",
   "metadata": {},
   "source": [
    "le pourcentage de valeur manquantes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "06c888b6-55af-40b4-bbeb-e1ba34786f67",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId     0.00\n",
       "Survived        0.00\n",
       "Pclass          0.00\n",
       "Name            0.00\n",
       "Sex             0.00\n",
       "Age            19.87\n",
       "SibSp           0.00\n",
       "Parch           0.00\n",
       "Ticket          0.00\n",
       "Fare            0.00\n",
       "Cabin          77.10\n",
       "Embarked        0.22\n",
       "dtype: float64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "round(df.isnull().sum()/df.shape[0],4).mul(100)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "9689321b-ff77-484d-9a24-b31e6d46b708",
   "metadata": {},
   "source": [
    "Le pourcentage de femmes ayant survecu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "377f3c58-c120-4cb1-aaea-d71a977ccb11",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived\n",
       "1    74.2\n",
       "0    25.8\n",
       "Name: proportion, dtype: float64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Survived[df[\"Sex\"]=='female'].value_counts(True).round(4).mul(100)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "9f5b44c9-db92-4f21-8534-69abb9a01996",
   "metadata": {},
   "source": [
    "Le pourcentage d'hommes ayant survecu"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8f501331-c70e-48e0-a9da-b8e5f384e710",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived\n",
       "0    81.11\n",
       "1    18.89\n",
       "Name: proportion, dtype: float64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Survived[df[\"Sex\"]=='male'].value_counts(True).round(4).mul(100)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "32555fcf-f71d-404d-926d-80249d854c22",
   "metadata": {},
   "source": [
    "La moyenne d'age par Pclass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "97b52374-3104-4639-a87b-045f3a04fce4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "29.6991"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.Age.mean().round(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "35b59e9b-b76c-4da2-a69a-524c4164d45f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.2615"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "round(df[(df[\"Sex\"]=='female')&(df.Survived==1)].shape[0]/df.shape[0],4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "18a3579b-2b0a-4e64-a91b-bbd107204984",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "81775fba-7f9f-4cdd-a536-fe293bc6810c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "raw",
   "id": "d2a57d83-d2ef-4af9-abf7-f08fb0f18014",
   "metadata": {},
   "source": [
    "Suppression des lignes de dataframe qui ont une valeur nulle dans les colonnes Age et Pclass\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "61b92189-329e-4d3a-a25b-0a8fc43355df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>22.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>26.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>35.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass   Age\n",
       "0         0       3  22.0\n",
       "1         1       1  38.0\n",
       "2         1       3  26.0\n",
       "3         1       1  35.0\n",
       "4         0       3  35.0"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# dropping irrelevant columns and rows with null values\n",
    "df1 = df.copy()\n",
    "df1 = df1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Sex', 'SibSp', 'Parch', 'Fare'], axis=1)\n",
    "df1 = df1.dropna(axis=0, how='any', subset=['Age', 'Pclass'])\n",
    "df1.head()"
   ]
  },
  {
   "cell_type": "raw",
   "id": "156575da-614e-4ee6-8785-58ed77e4f63f",
   "metadata": {},
   "source": [
    "Creation de dataframe X avec seulement les colonnes Age et Pclass\n",
    "Y avec seulement la colonne Survived\n",
    "Separation du jeu de donnees en train et test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "0cf0bf29-8539-4e11-999f-304d4e670b19",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(535, 2) (179, 2) (535,) (179,)\n"
     ]
    }
   ],
   "source": [
    "X = df1.drop('Survived', axis=1)\n",
    "y = df1['Survived']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\n",
    "X_train = X_train.reset_index().drop('index', axis=1) # resetting the index\n",
    "X_test = X_test.reset_index().drop('index', axis=1) # resetting the index\n",
    "print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "e56d60a5-925e-44a2-85d4-7ccc7092efba",
   "metadata": {},
   "source": [
    "utilisation du modele entaine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "0b151a15-b001-4f3f-a469-b0f4ba7a8ea8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training of the Machine Learning model (Random Forest)\n",
    "clf = RandomForestClassifier(max_depth=2, random_state=0)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# making prediction on the test set\n",
    "y_pred = clf.predict(X_test)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "6381ca0d-0188-49f0-b365-b4ec550d5084",
   "metadata": {},
   "source": [
    "Je vais mesurer le score de precision( en utilisant les metriques accuracy_score et f1_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "c33e32a3-bb4d-4356-b1ee-3aa985c5dae9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy score: 48.70%\n"
     ]
    }
   ],
   "source": [
    "print(f\"Accuracy score: {100*f1_score(y_test, y_pred):.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fd75d0bf-9400-4a65-8e2d-417814d1beaa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "556e1fcb-a5cb-4425-a62c-d898cc4e8c77",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(668, 11) (223, 11) (668,) (223,)\n"
     ]
    }
   ],
   "source": [
    "X = df.drop('Survived', axis=1)\n",
    "y = df['Survived']\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)\n",
    "X_train = X_train.reset_index().drop('index', axis=1) # resetting the index\n",
    "X_test = X_test.reset_index().drop('index', axis=1) # resetting the index\n",
    "print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "3dd57ba3-9019-4963-bb4a-848a3cb5071f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>299</td>\n",
       "      <td>1</td>\n",
       "      <td>Saalfeld, Mr. Adolphe</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>19988</td>\n",
       "      <td>30.5000</td>\n",
       "      <td>C106</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>885</td>\n",
       "      <td>3</td>\n",
       "      <td>Sutehall, Mr. Henry Jr</td>\n",
       "      <td>male</td>\n",
       "      <td>25.00</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>SOTON/OQ 392076</td>\n",
       "      <td>7.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>248</td>\n",
       "      <td>2</td>\n",
       "      <td>Hamalainen, Mrs. William (Anna)</td>\n",
       "      <td>female</td>\n",
       "      <td>24.00</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>250649</td>\n",
       "      <td>14.5000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>479</td>\n",
       "      <td>3</td>\n",
       "      <td>Karlsson, Mr. Nils August</td>\n",
       "      <td>male</td>\n",
       "      <td>22.00</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>350060</td>\n",
       "      <td>7.5208</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>306</td>\n",
       "      <td>1</td>\n",
       "      <td>Allison, Master. Hudson Trevor</td>\n",
       "      <td>male</td>\n",
       "      <td>0.92</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>113781</td>\n",
       "      <td>151.5500</td>\n",
       "      <td>C22 C26</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                             Name     Sex    Age  SibSp  \\\n",
       "0          299       1            Saalfeld, Mr. Adolphe    male    NaN      0   \n",
       "1          885       3           Sutehall, Mr. Henry Jr    male  25.00      0   \n",
       "2          248       2  Hamalainen, Mrs. William (Anna)  female  24.00      0   \n",
       "3          479       3        Karlsson, Mr. Nils August    male  22.00      0   \n",
       "4          306       1   Allison, Master. Hudson Trevor    male   0.92      1   \n",
       "\n",
       "   Parch           Ticket      Fare    Cabin Embarked  \n",
       "0      0            19988   30.5000     C106        S  \n",
       "1      0  SOTON/OQ 392076    7.0500      NaN        S  \n",
       "2      2           250649   14.5000      NaN        S  \n",
       "3      0           350060    7.5208      NaN        S  \n",
       "4      2           113781  151.5500  C22 C26        S  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "baca088d-a907-47fa-b425-a2105187987e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "raw",
   "id": "d564f73c-e1f2-47d0-9966-1a43a1d09e23",
   "metadata": {},
   "source": [
    "Importe la librairie OneHotEncode\n",
    "Instancie un objet OneHotEncoder\n",
    "Execute la fonction fit() sur les colonnes Sex et Embarked\n",
    "Execute la fonction transform() pour convertir les colonnes Sex et Embarked en colonnes numériques\n",
    "les nouvelles colonnes au dataframe initiale\n",
    "Supprime les colonnes d'origines\n",
    "Affichage du dataframe pour vérifier que les nouvelles colonnes ont bien été ajoutées\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "e39802b9-9447-49b0-a4f8-f653bada4e7d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Sex_female</th>\n",
       "      <th>Sex_male</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "      <th>Embarked_unknown</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>29.421343</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>30.5000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3</td>\n",
       "      <td>25.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.0500</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>24.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>14.5000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.5208</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>0.920000</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>151.5500</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass        Age  SibSp  Parch      Fare  Sex_female  Sex_male  \\\n",
       "0       1  29.421343      0      0   30.5000         0.0       1.0   \n",
       "1       3  25.000000      0      0    7.0500         0.0       1.0   \n",
       "2       2  24.000000      0      2   14.5000         1.0       0.0   \n",
       "3       3  22.000000      0      0    7.5208         0.0       1.0   \n",
       "4       1   0.920000      1      2  151.5500         0.0       1.0   \n",
       "\n",
       "   Embarked_C  Embarked_Q  Embarked_S  Embarked_unknown  \n",
       "0         0.0         0.0         1.0               0.0  \n",
       "1         0.0         0.0         1.0               0.0  \n",
       "2         0.0         0.0         1.0               0.0  \n",
       "3         0.0         0.0         1.0               0.0  \n",
       "4         0.0         0.0         1.0               0.0  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Train data preparation\n",
    "\n",
    "# dropping irrelevant columns\n",
    "X_train =  X_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)\n",
    "\n",
    "# replacing null values in Age and Embarked\n",
    "avg_age = X_train.Age.mean()\n",
    "X_train.Age =  X_train.Age.fillna(avg_age)\n",
    "X_train.Embarked = X_train.Embarked.fillna('unknown')\n",
    "\n",
    "# enconding categorical data using One Hot Encoding\n",
    "enc = OneHotEncoder(handle_unknown='ignore')\n",
    "enc.fit(X_train[['Sex', 'Embarked']])\n",
    "encoding = pd.DataFrame(enc.transform(X_train[['Sex', 'Embarked']]).toarray(), columns=enc.get_feature_names_out())\n",
    "X_train = X_train.join(encoding)\n",
    "X_train = X_train.drop(['Sex', 'Embarked'], axis=1)\n",
    "\n",
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "4a9ff3de-696f-466f-96d5-2520ee10587a",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Sex_female</th>\n",
       "      <th>Sex_male</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "      <th>Embarked_unknown</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>3</td>\n",
       "      <td>29.421343</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>15.2458</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10.5000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>33.0000</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>11.2417</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass        Age  SibSp  Parch     Fare  Sex_female  Sex_male  Embarked_C  \\\n",
       "0       3  29.421343      1      1  15.2458         0.0       1.0         1.0   \n",
       "1       2  31.000000      0      0  10.5000         0.0       1.0         0.0   \n",
       "2       3  20.000000      0      0   7.9250         0.0       1.0         0.0   \n",
       "3       2   6.000000      0      1  33.0000         1.0       0.0         0.0   \n",
       "4       3  14.000000      1      0  11.2417         1.0       0.0         1.0   \n",
       "\n",
       "   Embarked_Q  Embarked_S  Embarked_unknown  \n",
       "0         0.0         0.0               0.0  \n",
       "1         0.0         1.0               0.0  \n",
       "2         0.0         1.0               0.0  \n",
       "3         0.0         1.0               0.0  \n",
       "4         0.0         0.0               0.0  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Test data preparation\n",
    "\n",
    "# drop irrelevant columns\n",
    "X_test = X_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)\n",
    "\n",
    "# replacing null values in Age and Embarked\n",
    "X_test.Age =  X_test.Age.fillna(avg_age)\n",
    "X_test.Embarked = X_test.Embarked.fillna('unknown')\n",
    "\n",
    "# enconding categorical data using One Hot Encoding\n",
    "encoding = pd.DataFrame(enc.transform(X_test[['Sex', 'Embarked']]).toarray(), columns=enc.get_feature_names_out())\n",
    "X_test = X_test.join(encoding)\n",
    "X_test = X_test.drop(['Sex', 'Embarked'], axis=1)\n",
    "X_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f19bbf70-ef40-4ac6-b9de-518b05c89055",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "82342cbf-cbaf-4093-af17-101c8aed7314",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training of the Machine Learning model (Random Forest)\n",
    "clf = RandomForestClassifier(max_depth=2, random_state=0)\n",
    "clf.fit(X_train, y_train)\n",
    "\n",
    "# making prediction on the test set\n",
    "y_pred = clf.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "595f5e3c-3e19-466c-8d50-02d5f361d209",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy score: 72.41%\n"
     ]
    }
   ],
   "source": [
    "print(f\"Accuracy score: {100*f1_score(y_test, y_pred):.2f}%\")"
   ]
  },
  {
   "cell_type": "raw",
   "id": "bd1ff7b1-2369-4330-8d37-b4a88636b2ae",
   "metadata": {},
   "source": [
    "paramètre n_estimators (correspond au nombre d'arbre dans le random forest)\n",
    "\n",
    "- Entrainement de 7 modèles de RandomForest, avec n_estimators = 1, 10, 50, 100, 150, 200, 300\n",
    "- Pour chaque modèle, calcule la précision du modèle en utilisant la fonction f1_score()\n",
    "\n",
    " la valeur de n_estimators donnant la plus haute précision\n",
    "Objectif: tester différentes valeurs et voir comment ça impacte la précision du modèle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "75a152ce-a7ae-4a77-97cd-0591de7003eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy (n_estimators=1): 70.45%\n",
      "Accuracy (n_estimators=10): 72.51%\n",
      "Accuracy (n_estimators=50): 75.00%\n",
      "Accuracy (n_estimators=100): 75.00%\n",
      "Accuracy (n_estimators=150): 74.16%\n",
      "Accuracy (n_estimators=200): 75.00%\n",
      "Accuracy (n_estimators=300): 74.29%\n"
     ]
    }
   ],
   "source": [
    "## n_estimators\n",
    "for est in [1, 10, 50, 100, 150, 200, 300]:\n",
    "    clf = RandomForestClassifier(n_estimators=est, random_state=0)\n",
    "    clf.fit(X_train, y_train)\n",
    "\n",
    "    y_pred = clf.predict(X_test)\n",
    "\n",
    "    print(f\"Accuracy (n_estimators={est}): {100*f1_score(y_test, y_pred):.2f}%\")"
   ]
  },
  {
   "cell_type": "raw",
   "id": "8ac9b951-db28-4c11-b197-b60cc36faf25",
   "metadata": {},
   "source": [
    "paramètre max_depth (correspond à la profondeur maximum des arbres dans le random forest)\n",
    "\n",
    "- Entrainement de 5 modèles de RandomForest, avec max_depth = 1, 2, 10, 15, 20\n",
    "\n",
    "la valeur de max_depth donnant la plus haute précision?\n",
    "Objectif: tester différentes valeurs et voir comment ça impacte la précision du modèle\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "c181c6dc-5d7b-4652-a7e7-984835ff67c8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy (n_estimators=300): 71.43%\n",
      "Accuracy (n_estimators=300): 72.41%\n",
      "Accuracy (n_estimators=300): 76.30%\n",
      "Accuracy (n_estimators=300): 75.43%\n",
      "Accuracy (n_estimators=300): 75.00%\n"
     ]
    }
   ],
   "source": [
    "## max_depth\n",
    "for md in [1, 2, 10, 15, 20]:\n",
    "    clf = RandomForestClassifier(n_estimators=100, max_depth=md, random_state=0)\n",
    "    clf.fit(X_train, y_train)\n",
    "\n",
    "    y_pred = clf.predict(X_test)\n",
    "\n",
    "    print(f\"Accuracy (n_estimators={est}): {100*f1_score(y_test, y_pred):.2f}%\")"
   ]
  },
  {
   "cell_type": "raw",
   "id": "fee6a006-dd61-4a99-89d6-5297d0c06075",
   "metadata": {},
   "source": [
    "l'utilisation de Grid Search pour tester de manière plus efficace plusieurs valeurs de paramètres.\n",
    "\n",
    "But: tester différentes valeurs de paramètres et trouver la combinaison la plus optimale"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "03aa50e1-3adb-4914-b1e2-629556f16df6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# import GridSearchCV library\n",
    "from sklearn.model_selection import GridSearchCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "a8dbd25c-8901-4e26-9eed-1e1373a1fd00",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-3\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=10, estimator=RandomForestClassifier(random_state=0),\n",
       "             param_grid={&#x27;max_depth&#x27;: [1, 2, 10, 15, 20],\n",
       "                         &#x27;n_estimators&#x27;: [1, 10, 50, 100, 150, 200, 300]},\n",
       "             scoring=&#x27;accuracy&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-7\" type=\"checkbox\" ><label for=\"sk-estimator-id-7\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=10, estimator=RandomForestClassifier(random_state=0),\n",
       "             param_grid={&#x27;max_depth&#x27;: [1, 2, 10, 15, 20],\n",
       "                         &#x27;n_estimators&#x27;: [1, 10, 50, 100, 150, 200, 300]},\n",
       "             scoring=&#x27;accuracy&#x27;)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-8\" type=\"checkbox\" ><label for=\"sk-estimator-id-8\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=0)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-9\" type=\"checkbox\" ><label for=\"sk-estimator-id-9\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=0)</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=10, estimator=RandomForestClassifier(random_state=0),\n",
       "             param_grid={'max_depth': [1, 2, 10, 15, 20],\n",
       "                         'n_estimators': [1, 10, 50, 100, 150, 200, 300]},\n",
       "             scoring='accuracy')"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# fit # first we need to define the list of parameters and list of values we want to test\n",
    "# For each parameters, give a list of all values you want to test (same list used in previous exercise)\n",
    "parameters = {\n",
    "    'n_estimators':[1, 10, 50, 100, 150, 200, 300],\n",
    "    'max_depth':[1, 2, 10, 15, 20]\n",
    "}\n",
    "\n",
    "# init random forest object\n",
    "rf = RandomForestClassifier(random_state=0)\n",
    "# init grid search object\n",
    "gs = GridSearchCV(rf, param_grid  = parameters, cv=10, scoring='accuracy')\n",
    "# fit grid search object using train data\n",
    "gs.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "80a7caa0-47dc-46f4-84fb-adc896420255",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Best parameters: {'max_depth': 10, 'n_estimators': 200}\n"
     ]
    }
   ],
   "source": [
    "# printing the best set of parameters found by grid search\n",
    "print(f\"Best parameters: {gs.best_params_}\")\n",
    "\n",
    "# getting the trained model with best performance\n",
    "final_model = gs.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "698eff8b-5c94-462c-9376-5b3faf3dd524",
   "metadata": {},
   "outputs": [],
   "source": [
    "# making prediction on test set using best model\n",
    "y_pred = final_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "d2bfc6cb-233e-4538-a55c-7918d56474f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy (n_estimators=300): 75.14%\n"
     ]
    }
   ],
   "source": [
    " print(f\"Accuracy (n_estimators={est}): {100*f1_score(y_test, y_pred):.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "202f258e-2c35-470f-969d-f71f9bfed24e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1094622f-94a5-47ab-977c-4047acece7cf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
