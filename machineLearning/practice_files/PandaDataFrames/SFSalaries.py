import pandas as pd

sal = pd.read_csv('Salaries.csv')
sal.head()
print('salaries info: ', sal.info())
averageBasePay = sal['BasePay'].mean()
print('average base pay: ', averageBasePay)
maxOvertime = sal['OvertimePay'].max()
print('max overtime pay: ', maxOvertime)

joesJob = sal[sal['EmployeeName'] == 'JOSEPH DRISCOL']['JobTitle']
print('JOESPH DRISCOLL\'s job title:', joesJob)

joesTotalWage = sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']
highestPaidEmployee = sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]['EmployeeName']
highestPaidEmployeeInfo = sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].max()]
lowestPaidEmployee = sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]

print("Joes total wage: ", joesTotalWage)
print("highest paid person: ", highestPaidEmployee)
print('highest paid emplyee info: \n', highestPaidEmployeeInfo)
print('+---================---+')
print('lowest paid employee info: \n', lowestPaidEmployee)


meanBasePay = sal.groupby('Year').mean()['BasePay']

print("mean base pay over 4 years:\n", meanBasePay)

uniqueJobsCount = len(sal['JobTitle'].unique())
# uniqueJobsCount = len((sal['JobTitle'].value_counts() == 1)) will also work
print('number of unique jobs: ', uniqueJobsCount)

top5CommonJobs = sal['JobTitle'].value_counts()[:5]
# value_counts automatically orders list with greatest value at index 0
print('+---================---+')
print('top 5 most common jobs:\n', top5CommonJobs)

#oneJobCount = len(sal[sal['Year'] == 2013]['JobTitle'].unique()) will not work
oneJobCount = sum(sal[sal['Year'] == 2013]['JobTitle'].value_counts() == 1)
print('+---================---+')
print('list on jobs with only one count in 2013:\n',oneJobCount)

def findCheif(job):
    if 'chief' in job.lower():
        return True
    else:
        return False

cheifCount = sum(sal['JobTitle'].apply(lambda x: findCheif(x)))
print('+---================---+')
print('number of jobs with chief in title: ', cheifCount)

print('+---================---+')
print("looking for correlation between length of the Job Title string and Salary:\n", )
sal['title_len'] = sal['JobTitle'].apply(len)
print(sal[['title_len','TotalPayBenefits']].corr()) # no Correlation close to 1:1 is a correlation
