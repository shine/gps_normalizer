import copy
import numpy as np
import matplotlib.pyplot as plt

class GpsNormalizer:
    def __init__(self, json):
        self.initial_json = copy.deepcopy(json)
        self.final_json = copy.deepcopy(json)

        self.was_modified = False

    def apply_default(self):
        '''Apply recommended method of normalization'''
        return self.apply_spc_and_average()

    def apply_spc_and_average(self):
        '''This method will apply statistical process control to initial data
        and all anomalies will be solved by replacement with mean value'''
        self.add_distances(self.final_json)
        self.__fix_anomaly_by_average(self.__find_anomaly_indexes_by_spc())
        self.add_distances(self.final_json) # regenerate distances after coordinates update

        return self.__result()

    def show_distribution(self, json):
        '''Method will plot distances distribution'''
        array = [item['Distance'] for item in json if 'Distance' in item.keys()]
        array = np.absolute(array)
        plt.plot(list(range(1, len(array)+1)), array, 'ro')
        plt.show()

        return self

    def __result(self):
        '''Returns initial JSON or its modified version'''
        if self.was_modified:
            return self.keep_coordinates(self.final_json)
        return self.keep_coordinates(self.initial_json)

    def keep_coordinates(self, json):
        '''Filter out all information except latitude and longitude'''
        return [{'Latitude': item['Latitude'], 'Longitude': item['Longitude']} for item in json]

    def add_distances(self, json):
        '''Complexity: O(n)'''
        for index, item in enumerate(json):
            if index > 0:
                prev_lat = json[index-1]['Latitude']
                prev_long = json[index-1]['Longitude']
                curr_lat = item['Latitude']
                curr_long = item['Longitude']

                json[index]['Distance'] = np.sqrt((curr_lat-prev_lat)**2+(curr_long-prev_long)**2)

        return self

    def __find_anomaly_indexes_by_spc(self):
        '''Complexity: O(2n)'''
        distances = [item['Distance'] for item in self.final_json if 'Distance' in item.keys()]
        limit = np.mean(distances) + np.std(distances)
        result = []

        for index, item in enumerate(distances):
            if item > limit:
                result.append(index)

        return result

    def __fix_anomaly_by_average(self, anomaly_indexes):
        '''Complexity: O(k), k << n'''
        for index in anomaly_indexes:
            if index-1 in anomaly_indexes:
                prev_lat = self.final_json[index-1]['Latitude']
                prev_long = self.final_json[index-1]['Longitude']
                next_lat = self.final_json[index+1]['Latitude']
                next_long = self.final_json[index+1]['Longitude']

                self.final_json[index]['Latitude'] = np.mean([prev_lat, next_lat])
                self.final_json[index]['Longitude'] = np.mean([prev_long, next_long])

                self.was_modified = True

        return self
