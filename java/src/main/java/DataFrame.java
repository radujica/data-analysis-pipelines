import java.util.HashMap;
import java.util.Map;

public class DataFrame {
    private final Map<String, Object> data;

    DataFrame() {
        this.data = new HashMap<>();
    }

    void add(String name, Object data) {
        this.data.put(name, data);
    }

    Object get(String name) {
        return this.data.get(name);
    }

    void put(String name, Object data) {
        this.data.put(name, data);
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
        float[] newLat = new float[newSize];
        times = lon.length;
        index = 0;
        for (int i = 0; i < times; i++) {
            for (float aLat : lat) {
                newLat[index] = aLat;
                index++;
            }
        }
        times = tim.length;
        index = 0;
        for (float aLat : lat) {
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

    private int[] filterTime(String columnName, boolean[] indexes, int newSize) {
        int[] column = (int[]) this.get(columnName);
        int[] newColumn = new int[newSize];
        int tgi = 0;
        for (int i = 0; i < column.length; i++) {
            if (indexes[i]) {
                newColumn[tgi] = column[i];
                tgi++;
            }
        }

        return newColumn;
    }

    private float[] filterColumn(String columnName, boolean[] indexes, int newSize) {
        float[] column = (float[]) this.get(columnName);
        float[] newColumn = new float[newSize];
        int tgi = 0;
        for (int i = 0; i < column.length; i++) {
            if (indexes[i]) {
                newColumn[tgi] = column[i];
                tgi++;
            }
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

        DataFrame df = new DataFrame();

        String[] df1Cols = {"tg", "tg_err", "pp", "pp_err", "rr", "rr_err"};
        String[] df2Cols = {"tn", "tn_err", "tx", "tx_err"};

        df.put(joinOn[0], this.filterColumn(joinOn[0], df1NewIndex, newSize));
        df.put(joinOn[1], this.filterColumn(joinOn[1], df1NewIndex, newSize));
        df.put(joinOn[2], this.filterTime(joinOn[2], df1NewIndex, newSize));

        for (String columnName : df1Cols) {
            df.put(columnName, this.filterColumn(columnName, df1NewIndex, newSize));
        }

        for (String columnName : df2Cols) {
            df.put(columnName, df2.filterColumn(columnName, df2NewIndex, newSize));
        }

        return df;
    }
}
