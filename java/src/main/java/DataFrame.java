import ucar.nc2.time.CalendarDate;

import java.util.*;

public class DataFrame {
    private final Map<String, Object> data;
    private final int size;

    DataFrame() {
        this.data = new HashMap<>();
        this.size = 0;
    }

    DataFrame(int size) {
        this.data = new HashMap<>();
        this.size = size;
    }

    void add(String name, Object data) {
        this.data.put(name, data);
    }

    Map<String, Object> values() {
        return this.data;
    }

    Object get(String name) {
        return this.data.get(name);
    }

    void put(String name, Object data) {
        this.data.put(name, data);
    }

    Set<String> keys() {
        return this.data.keySet();
    }

    void pop(String columnName){
        this.data.remove(columnName);
    }

    int getSize() {
        return this.size;
    }

    private void cartesianProductDimensions() {
        String[] dims = {"longitude", "latitude", "time"};

        float[] lon = (float[]) this.get(dims[0]);
        float[] lat = (float[]) this.get(dims[1]);
        int[] tim = (int[]) this.get(dims[2]);
        int newSize = lon.length * lat.length * tim.length;

        // inner x2
        float[] newLon = new float[newSize];
        int times = lat.length * tim.length;
        int index = 0;
        for (float aLon : lon) {
            for (int j = 0; j < times; j++) {
                newLon[index] = aLon;
                index++;
            }
        }

        // outer + inner
        float[] newLatTemp = new float[lon.length * lat.length];
        times = lon.length;
        index = 0;
        for (int i = 0; i < times; i++) {
            for (float aLat : lat) {
                newLatTemp[index] = aLat;
                index++;
            }
        }
        float[] newLat = new float[newSize];
        times = tim.length;
        index = 0;
        for (float aLat : newLatTemp) {
            for (int j = 0; j < times; j++) {
                newLat[index] = aLat;
                index++;
            }
        }

        // outer x2
        int[] newTim = new int[newSize];
        times = lon.length * lat.length;
        index = 0;
        for (int i = 0; i < times; i++) {
            for (int aTim : tim) {
                newTim[index] = aTim;
                index++;
            }
        }

        this.put(dims[0], newLon);
        this.put(dims[1], newLat);
        this.put(dims[2], newTim);
    }

    private Object filterColumn(String columnName, boolean[] indexes, int newSize) {
        Object data = this.get(columnName);
        Object newColumn;
        if (data.getClass().equals(float[].class)) {
            float[] column = (float[]) this.get(columnName);
            float[] newData = new float[newSize];
            int tgi = 0;
            for (int i = 0; i < column.length; i++) {
                if (indexes[i]) {
                    newData[tgi] = column[i];
                    tgi++;
                }
            }

            newColumn = newData;
        } else if (data.getClass().equals(int[].class)) {
            int[] column = (int[]) this.get(columnName);
            int[] newData = new int[newSize];
            int tgi = 0;
            for (int i = 0; i < column.length; i++) {
                if (indexes[i]) {
                    newData[tgi] = column[i];
                    tgi++;
                }
            }

            newColumn = newData;
        } else {
            throw new RuntimeException("neither float nor int data!");
        }

        return newColumn;
    }

    DataFrame join(DataFrame df2) {
        String[] joinOn = {"longitude", "latitude", "time"};

        this.cartesianProductDimensions();
        df2.cartesianProductDimensions();

        float[] lon1 = (float[]) this.get(joinOn[0]);
        float[] lon2 = (float[]) df2.get(joinOn[0]);
        float[] lat1 = (float[]) this.get(joinOn[1]);
        float[] lat2 = (float[]) df2.get(joinOn[1]);
        int[] tim1 = (int[]) this.get(joinOn[2]);
        int[] tim2 = (int[]) df2.get(joinOn[2]);

        int totalSize1 = ((float[]) this.get(joinOn[0])).length;
        int totalSize2 = ((float[]) df2.get(joinOn[0])).length;
        int index1 = 0, index2 = 0;

        boolean[] df1NewIndex = new boolean[totalSize1];
        boolean[] df2NewIndex = new boolean[totalSize2];
        int newSize = 0;

        while (index1 < totalSize1 && index2 < totalSize2) {
            float lo1 = lon1[index1];
            float lo2 = lon2[index2];

            if (lo1 == lo2) {
                float la1 = lat1[index1];
                float la2 = lat2[index2];

                if (la1 == la2) {
                    int ti1 = tim1[index1];
                    int ti2 = tim2[index2];

                    if (ti1 == ti2) {
                        df1NewIndex[index1] = true;
                        df2NewIndex[index2] = true;

                        index1++;
                        index2++;
                        newSize++;
                    } else {
                        if (ti1 < ti2) {
                            df1NewIndex[index1] = false;
                            index1++;
                        } else {
                            df2NewIndex[index2] = false;
                            index2++;
                        }
                    }
                } else {
                    if (la1 < la2) {
                        df1NewIndex[index1] = false;
                        index1++;
                    } else {
                        df2NewIndex[index2] = false;
                        index2++;
                    }
                }
            } else {
                if (lo1 < lo2) {
                    df1NewIndex[index1] = false;
                    index1++;
                } else {
                    df2NewIndex[index2] = false;
                    index2++;
                }
            }
        }

        while (index1 < totalSize1) {
            df1NewIndex[index1] = false;
            index1++;
        }

        while (index2 < totalSize2) {
            df2NewIndex[index2] = false;
            index2++;
        }

        DataFrame df = new DataFrame(newSize);

        String[] df1Cols = {"tg", "tg_stderr", "pp", "pp_stderr", "rr", "rr_stderr"};
        String[] df2Cols = {"tn", "tn_stderr", "tx", "tx_stderr"};

        for (String columnName : joinOn) {
            df.put(columnName, this.filterColumn(columnName, df1NewIndex, newSize));
        }

        for (String columnName : df1Cols) {
            df.put(columnName, this.filterColumn(columnName, df1NewIndex, newSize));
        }

        for (String columnName : df2Cols) {
            df.put(columnName, df2.filterColumn(columnName, df2NewIndex, newSize));
        }

        return df;
    }

    // Time kept as original int so use this method to convert to string.
    // Note both datasets have the same starting date, otherwise the join would require
    // conversion before
    static String intTimeToString(int time, String calendar, String units) {
        String udunits = String.valueOf(time) + " " + units;

        return CalendarDate.parseUdunits(calendar, udunits).toString().substring(0, 10);
    }

    DataFrame subset(int start, int end) {
        int numberRows = end - start;
        DataFrame subsetData = new DataFrame(numberRows);

        for (String column : this.keys()) {
            Object data = this.get(column);

            if (data.getClass().equals(float[].class)) {
                float[] dataRaw = (float[]) data;
                float[] subset = new float[numberRows];
                System.arraycopy(dataRaw, start, subset, 0, numberRows);

                subsetData.put(column, subset);
            // only time is int
            } else if (data.getClass().equals(int[].class)) {
                int[] dataRaw = (int[]) data;
                int[] subset = new int[numberRows];
                System.arraycopy(dataRaw, start, subset, 0, numberRows);

                subsetData.put(column, subset);
            }
        }

        return subsetData;
    }

    DataFrame filter() {
        boolean[] finalFilter = new boolean[this.size];
        float[] rawTg = (float[]) this.get("tg");
        float[] rawPp = (float[]) this.get("pp");
        float[] rawRr = (float[]) this.get("rr");
        float tgFilter = -99.99f;
        float ppFilter = -999.9f;
        float rrFilter = -999.9f;
        int newSize = 0;

        for (int i = 0; i < this.size; i++) {
            boolean toKeep = rawTg[i] != tgFilter & rawPp[i] != ppFilter & rawRr[i] != rrFilter;
            if (toKeep) {
                newSize++;
            }
            finalFilter[i] = toKeep;
        }

        DataFrame newDf = new DataFrame(newSize);
        for (String columnName : this.keys()) {
            newDf.put(columnName, filterColumn(columnName, finalFilter, newSize));
        }

        return newDf;
    }

    DataFrame aggregations() {
        int numberAggregations = 4;
        DataFrame df = new DataFrame(numberAggregations);

        // make sure the keys are copied to not remove from dataset
        Set<String> keys = new HashSet<>(this.keys());
        keys.removeAll(Arrays.asList("longitude", "latitude", "time"));

        df.put("agg", new String[]{"min", "max", "mean", "std"});

        for (String columnName : keys) {
            float[] aggregations = new float[numberAggregations];

            float[] data = (float[]) this.get(columnName);

            float max = Float.MIN_VALUE;
            float min = Float.MAX_VALUE;
            float sum = 0f;
            for (float number : data) {
                if (number > max) {
                    max = number;
                }
                if (number < min) {
                    min = number;
                }
                sum += number;
            }
            aggregations[0] = min;
            aggregations[1] = max;

            float mean = sum / ((float) data.length);

            float numerator = 0;
            for (float number : data) {
                numerator += Math.pow(number - mean, 2);
            }
            float std = (float) Math.sqrt(numerator / ((float) (data.length - 1)));

            aggregations[2] = mean;
            aggregations[3] = std;

            df.put(columnName, aggregations);
        }

        return df;
    }

    private double computeMean(float[] columnData, List<Integer> groupMembers) {
        double sum = 0;
        for (int i : groupMembers) {
            sum += columnData[i];
        }
        return sum / groupMembers.size();
    }

    private class Key {
        // key to group on
        private final float lon;
        private final float lat;
        private final String tim;

        Key(float lon, float lat, String tim) {
            this.lon = lon;
            this.lat = lat;
            this.tim = tim;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null) return false;
            if (!(o instanceof Key)) return false;

            Key other = (Key) o;

            return this.tim.equals(other.tim) &&
                    this.lat == other.lat &&
                    this.lon == other.lon;
        }

        @Override
        public int hashCode() {
             return Objects.hash(this.tim, this.lat, this.lon);
        }
    }

    DataFrame groupBy() {
        DataFrame result = new DataFrame(this.size);

        String[] groupOn = {"longitude", "latitude", "year_month"};
        String[] aggregateOn = {"tg", "tn", "tx", "pp", "rr"};

        float[] lon = (float[]) this.get(groupOn[0]);
        float[] lat = (float[]) this.get(groupOn[1]);
        String[] tim = (String[]) this.get(groupOn[2]);

        float[] tg = (float[]) this.get(aggregateOn[0]);
        float[] tn = (float[]) this.get(aggregateOn[1]);
        float[] tx = (float[]) this.get(aggregateOn[2]);
        float[] pp = (float[]) this.get(aggregateOn[3]);
        float[] rr = (float[]) this.get(aggregateOn[4]);

        Map<Key, List<Integer>> groups = new HashMap<>();

        // same length
        for (int i = 0; i < lon.length; i++) {
            Key key = new Key(lon[i], lat[i], tim[i]);
            if (groups.containsKey(key)) {
                List<Integer> groupMembers = groups.get(key);
                groupMembers.add(i);
            } else {
                List<Integer> newGroup = new ArrayList<>();
                newGroup.add(i);
                groups.put(key, newGroup);
            }
        }

        // can already store the means in the new column; fake join
        double[] meansTg = new double[groups.size()];
        double[] meansTn = new double[groups.size()];
        double[] meansTx = new double[groups.size()];
        double[] meansPp = new double[groups.size()];
        double[] meansRr = new double[groups.size()];
        int i = 0;
        for (Map.Entry entry : groups.entrySet()) {
            List<Integer> groupMembers = (List<Integer>) entry.getValue();
            double mean = computeMean(tg, groupMembers);
            meansTg[i] = mean;

            mean = computeMean(tn, groupMembers);
            meansTn[i] = mean;

            mean = computeMean(tx, groupMembers);
            meansTx[i] = mean;

            mean = computeMean(pp, groupMembers);
            meansPp[i] = mean;

            mean = computeMean(rr, groupMembers);
            meansRr[i] = mean;

            i++;
        }
        result.put("tg_mean", meansTg);
        result.put("tn_mean", meansTn);
        result.put("tx_mean", meansTx);
        result.put("pp_mean", meansPp);
        result.put("rr_mean", meansRr);

        return result;
    }

    DataFrame sum() {
        DataFrame result = new DataFrame(this.data.size());

        String[] aggregateOn = {"tg_mean", "tn_mean", "tx_mean", "pp_mean", "rr_mean"};

        double[] tg = (double[]) this.get(aggregateOn[0]);
        double[] tn = (double[]) this.get(aggregateOn[1]);
        double[] tx = (double[]) this.get(aggregateOn[2]);
        double[] pp = (double[]) this.get(aggregateOn[3]);
        double[] rr = (double[]) this.get(aggregateOn[4]);

        double[] sums = new double[this.data.size()];
        double sum = 0;
        for (double aTg : tg) {
            sum += aTg;
        }
        sums[0] = sum;
        sum = 0;
        for (double aTn : tn) {
            sum += aTn;
        }
        sums[1] = sum;
        sum = 0;
        for (double aTx : tx) {
            sum += aTx;
        }
        sums[2] = sum;
        sum = 0;
        for (double aPp : pp) {
            sum += aPp;
        }
        sums[3] = sum;
        sum = 0;
        for (double aRr : rr) {
            sum += aRr;
        }
        sums[4] = sum;

        result.put("column", aggregateOn);
        result.put("grouped_sum", sums);

        return result;
    }
}
