import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)


    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))


    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """

        row_num = len(self.data) # gets the number of lists in the self.data list (rows)
        col_num = len(self.column_names) # gets the number of columns

        return row_num, col_num


    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
            IndexError: if col_identifier is not a valid index value
        """
        return_column = []

        if type(col_identifier) is str: # checks if the value is a string
            try:
                col_identifier = self.column_names.index(col_identifier) # converts the column name to an index for each row
            except ValueError:
                print("ERROR: the column is not in the data file") # if the string given is not in the column
                return return_column
        
        try: # if the col_identifier is an index
            for row in self.data:
                if (not include_missing_values and row[col_identifier] == "NA"): # checks if user does not want to include missing values, and a row in the column has an "NA"
                    pass
                else: 
                    return_column.append(row[col_identifier])
        except IndexError:
            print("ERROR: this index is not in the column indexes") # if the index given is out of scope

        return return_column


    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Raises:
            ValueError: if vale cannot be a number (is a NA value)

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """

        for row in self.data:
            for index in range(len(row)):
                try:
                    row[index] = float(row[index])
                except ValueError: # if the value is NA
                    pass


    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """

        row_indexes_to_drop.sort(reverse = True) # sorts the values in descending order, since if we delete starting from the top of the table, everything shifts up, and we lose the indexes of which values we are deleting 

        iterator = len(self.data) - 1 # starts at the bottom of the table
        row_index = 0

        while iterator >= 0 and row_index < len(row_indexes_to_drop): # checks we are not at the top of the table (since we're starting at the bottom) and we have dropped all values in the list given
            if iterator == row_indexes_to_drop[row_index]:
                del self.data[iterator]
                row_index += 1
            iterator -= 1
            

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, "r") as in_file: # "r" reads the values
            contents = csv.reader(in_file)

            for content in contents:
                self.data.append(content) # saves all the data from the file
            
            self.column_names = self.data.pop(0) # removes (and saves) the first row as the column

            self.convert_to_numeric() # immediately converts values to numeric

        return self


    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """

        with open (filename, "w") as out_file:
    
            
            out_file = csv.writer(out_file)
            out_file.writerow(self.column_names) # writes the header
            out_file.writerows(self.data) # writes all the data


    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """

        duplicate_indexes = []
        key_column_index = []

        for value in key_column_names:
            key_column_index.append(self.column_names.index(value))

        for row_index, row in enumerate(self.data):
            curr_row = row_index # sets the first row it finds to "curr_row", so it can check the other rows without having to move off of the current one (also avoids the current row from being seen as a duplicate)
            
            while curr_row < len(self.data):
                is_same = True
                
                for index in key_column_index:
                    if self.data[curr_row][index] != row[index]: # checks if the saved row's value is (not) the same as the row we're looking at (as we're going down the table)'s value
                        is_same = False

                if is_same: # if the value in the columns are the same, it gets added to the duplicate indexes
                    if(curr_row != row_index and curr_row not in duplicate_indexes): # ensures the rows that are duplicates that are saved aren't aleady in the list (since we can have more than 2 duplicate rows)
                        duplicate_indexes.append(curr_row)

                curr_row += 1
            

        duplicate_indexes.sort() # sorts the rows in order

        return duplicate_indexes


    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        
        missing_values_index = []

        for row_index, row in enumerate(self.data):
            for index in row: # checks each column of each row
                if index == "NA":
                    if row_index not in missing_values_index:
                        missing_values_index.append(row_index) # saves all the rows that has a missing value, so it can call the drop_rows method

        self.drop_rows(missing_values_index)
        

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Raises:
            TypeError: if value is NA

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """

        col_index = self.column_names.index(col_name) # finds the index of the column name

        total_values = 0
        value_sum = 0

        for row in self.data:
            try: 
                value_sum += row[col_index] # adds all the values up in each column
                total_values += 1
            except TypeError: # if the value is "NA"
                pass

        new_value = float(value_sum // total_values) # taking floor value, since all values in sample data (auto-mpg.txt) have 0's after decimal places (like an integer)
        
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = new_value # sets all the "NA" values to be the average

   

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Raises:
            ValueError: if NA value is added/treated as a numerical value

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """        
        
        col_indexes = []

        for value in col_names:
            col_indexes.append(self.column_names.index(value)) # saves all the indexes that correspond to the column names we want to get info about

        new_header = []
        new_data = []

        if len(self.data) == 0: # if the data is 0, the program crashes (since we may accidentally divide by 0 if we wanted to find the mean), so this returns beforehand
            new_data = [0,0,0,0,0]
            return MyPyTable(new_data, new_header)

        # calculates (and stores) the min and max value of each column into a list
        min_values = [0] * len(col_indexes)
        max_values = [0] * len(col_indexes)

        for value in range(len(min_values)):
            min_values[value] = float(self.data[0][col_indexes[value]]) # sets the initial values to be the first row. If it was set to an actual number, the number we set it to may be the lowest (even though it may not be in the dataset)
            max_values[value] = float(self.data[0][col_indexes[value]]) # also, all the values are saved to a list, so each item in the list represents a column

        for row in self.data:
            for index, index_value in enumerate(col_indexes):
                try:
                    if min_values[index] > float(row[index_value]): # checks each value in each row for each column
                        min_values[index] = float(row[index_value])
                    if max_values[index] < float(row[index_value]):
                        max_values[index] = float(row[index_value])
                except ValueError: # in the case the row has "NA", this ignores the data
                    pass 

        # calculates (and stores) the mid of each column into a list
        mid_values = [0] * len(col_indexes)

        for index in range(len(mid_values)):
            mid_values[index] = round(float((min_values[index] + max_values[index]) / 2), 2) # ensures we round to 2 decimals

        # calculates (and stores) the average of each column into a list
        avg_values = [0] * len(col_indexes)
        values_num = [0] * len(col_indexes)

        for iterator, col in enumerate(col_indexes):
            for row in self.data:
                if type(row[col]) is not str: # in case a row contains "NA"
                    avg_values[iterator] += row[col] # adds the total values of each row (so it later gets divided)
                    values_num[iterator] += 1 # counts the number of rows with valid data (so it later divides the total)
                
        for index, values in enumerate(avg_values):
            avg_values[index] = float(values / values_num[index]) # stores the values of averages in a list with each index being a column name given
        
        # calculates (and stores) the median of each column into a list
        median_values = [0] * len(col_indexes)

        for iterator, col in enumerate(col_indexes):
            all_values = []
            for row in self.data:
                if type(row[col]) is not str:
                    all_values.append(row[col]) # adds all the data from each column into a list, since we need values from each row before computing the median
            all_values.sort() # to find median, all values need to be sorted from small --> large
            if len(all_values) % 2 == 1: # if the list has an odd number of values, the median is just the middle
                median_values[iterator] = float(all_values[(len(all_values)//2)])
            else: # if the list is odd
                middle = len(all_values) // 2 # finds the lower of the 2 middle values
                middle_value = round(float((all_values[middle] + all_values[middle - 1])/2), 2) # divides the 2 upper and lower middle values
                median_values[iterator] = middle_value
        
        # inputs all data into a MyPyTable object
        for index in range(len(col_names)):
            insertion_list = []
            insertion_list.append(col_names[index])
            insertion_list.append(min_values[index])
            insertion_list.append(max_values[index])
            insertion_list.append(mid_values[index])
            insertion_list.append(avg_values[index])
            insertion_list.append(median_values[index])

            new_data.append(insertion_list)

        new_header = ["attribute", "min", "max", "mid", "avg", "median"]

        summary_stats = MyPyTable(new_header, new_data)
        return summary_stats

    def get_columns(self, col_identifiers, include_missing_values = True):
            """Extracts multiple columns from the table data as a 2D list

            Parameters:
                col_identifiers (list of str or int): string for column names or int
                    for column indices
                include_missing_values (bool): True if missing values ("NA") should be
                    included in any of the returning columns, False otherwise.

            Returns:
                list of list of obj: 2D list of values in all the specified columns
            
            Raises:
                ValueError: if col_identifier is invalid
                IndexError: if col_identifier is not a valid index value
            """
            
            return_columns = []
            col_indexes = []

            if type(col_identifiers[0]) is str: # checks if the given columns are in str or int (col name or index of columns)
                try:
                    for col_name in col_identifiers:
                        col_indexes.append(self.column_names.index(col_name)) # adds the column value for a row into a list, so the resulting list is 2D
                except ValueError:
                    print("ERROR: not all columns given are in the data file") # if the column string given is not in the column
            
            col_indexes.sort() # in case a user inputs out-of-order indices

            try:
                for row in self.data:
                    curr_row = []    
                    if(not include_missing_values and "NA" in row): # checks if user does not want to include missing values, and a row in a given column has an "NA"
                        pass
                    else:
                        for index, value in enumerate(row):
                            if index in col_indexes:
                                curr_row.append(value)
                    return_columns.append(curr_row)
            except IndexError:
                print("ERROR: this index is not in the column indexes") # if the index is out of scope

            return return_columns

    def normalize_columns(self, column_names):
        """Min-max normalizes all given continuous columns 

        Parameters:
            column_names (list of string): list of column names that will be scaled

        Notes:
            Assume column_names are exist in table and are all numeric values

            This function calls the compute_summary_statistics column (same class function)
        
        """
        stats = self.compute_summary_statistics(column_names) 

        range = []
        for col in stats.data:
            range.append(col[2] - col[1]) # computes the range from the max and min value (used as denominator)
            
        column_indexes = []
        for col_name in column_names:
            column_indexes.append(self.column_names.index(col_name)) # saves the indexes of column names

        for row in self.data:
            for position, col_index in enumerate(column_indexes):
                row[col_index] = round((row[col_index] - stats.data[position][1]) / range[position], 2) # sets values as scaled value