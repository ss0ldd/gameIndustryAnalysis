import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import shapiro
from scipy.stats import wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def data_info(data):
    print(data[0:3])

    print("\nИнформация")
    print(data.info())
    print("\nНулевые строки")
    print(data.isnull().sum())

    print("\nХарактеристики данных")
    print(data.describe())

    print("\nКорреляции")
    print(data.corr(numeric_only=True))

def seaborn(data, column):
    print("\nГистограмма")
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, bins=30, color="skyblue", stat="density")
    plt.title(column)
    plt.grid(True)
    plt.show()

def boxplot(data, column):
    print("\nБоксплот")
    plt.boxplot(x=data[column])
    plt.title(f"{column} boxplot")
    plt.xlabel(column)
    Q1 = data[column].quantile(0.25)
    Q2 = data[column].quantile(0.5)
    Q3 = data[column].quantile(0.75)

    plt.text(1.1, Q1, f'Q1: {Q1:.2f}', ha='center', va='center')
    plt.text(1.1, Q2, f'Median: {Q2:.2f}', ha='center', va='center')
    plt.text(1.1, Q3, f'Q3: {Q3:.2f}', ha='center', va='center')
    plt.show()

def scatterplot(data, x, y):
    print("\nДиаграмма Рассеивания")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x, y=y)
    plt.title("Зависимость установок от цены")
    plt.show()

def outliers_counter(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
        outliers = data[(data[column] < lowerBound) | (data[column] > upperBound)]
        print(f"Выбросы по столбцу '{column}': ", len(outliers))


def shapiro_test(data, columns):
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        cleaned_column = data[column].dropna()

        stat, p = shapiro(cleaned_column)
        alpha = 0.05
        print(f"\nСтолбец: '{column}'")
        if p > alpha:
            print("Распределение нормальное (не отвергаем H₀)")
        else:
            print("Распределение не нормальное (отвергаем H₀)")

def hypothesis_test(data, column, hypothesized_median):
    data[column] = pd.to_numeric(data[column], errors='coerce')
    cleaned = data[column].dropna()
    wilcoxon_stat, wilcoxon_p = wilcoxon(cleaned - hypothesized_median)
    print(f"\nТест Уилкоксона для столбца '{column}'")
    print(f"Гипотетическая медиана: {hypothesized_median}")
    print(f"p-value: {wilcoxon_p:.4f}")
    if wilcoxon_p < 0.05:
        print("Вывод: Отвергаем нулевую гипотезу — медиана отличается от гипотетической.")
    else:
        print("Вывод: Не отвергаем нулевую гипотезу — нет оснований считать медиану отличной.")


def linear_regression(data, x_column, y_column):
    data[x_column] = pd.to_numeric(data[x_column], errors='coerce')
    data[y_column] = pd.to_numeric(data[y_column], errors='coerce')

    cleaned_data = data[[x_column, y_column]].dropna()

    X = cleaned_data[[x_column]]
    y = cleaned_data[y_column]

    # Логарифмируем (добавляем 1, чтобы избежать log(0))
    logged_X = np.log1p(X)
    logged_y = np.log1p(y)

    model = LinearRegression()
    model.fit(logged_X, logged_y)
    y_pred = model.predict(logged_X)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(logged_y, y_pred)

    print(f"\nЛинейная регрессия: {y_column} ~ {x_column}")
    print(f"Уравнение: log({y_column}) = {slope:.4f} * log({x_column}) + {intercept:.4f}")
    print(f"R²: {r2:.4f}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=cleaned_data[x_column], y=cleaned_data[y_column], alpha=0.7, label='Данные')
    sns.lineplot(x=cleaned_data[x_column], y=np.expm1(y_pred), color='red', label='Регрессия (обратное преобразование)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Линейная регрессия: {y_column} от {x_column} (логарифмический масштаб)')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    filePath = 'bestSelling_games.csv'
    data = pd.read_csv(filePath)

    data_info(data)

    seaborn(data, 'price')

    boxplot(data, 'rating')

    scatterplot(data, 'price', 'estimated_downloads')

    nonNumericColumns = data.select_dtypes(include=[np.number]).columns.tolist()

    #Выброс по одному столбцу
    outliers_counter(data, ['price'])

    # Выбросы по всем столбцам
    # outliers_counter(data, nonNumericColumns)

    # Тест Шапиро и гипотеза по одному столбцу
    shapiro_test(data, ['price'])
    hypothesis_test(data, 'price', hypothesized_median=10)

    # Тест Шапиро и гипотеза по всем столбцам
    # shapiro_test(data, nonNumericColumns)
    # for column in nonNumericColumns:
    #     hypothesis_test(data, column, hypothesized_median=10)

    linear_regression(data, 'estimated_downloads', 'price')


if __name__ == '__main__':
    main()
