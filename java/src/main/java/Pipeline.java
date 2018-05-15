import com.google.common.base.Joiner;
import com.google.common.primitives.Floats;
import ucar.nc2.Attribute;
import ucar.nc2.NetcdfFile;
import ucar.nc2.Variable;

import java.io.IOException;
import java.util.Map;

// TODO: convert for loops to streams? no native support for floats though
// https://stackoverflow.com/questions/26951431/how-to-get-an-observablefloatarray-from-a-stream/26970398#26970398
public class Pipeline  {
    private static final String PATH = System.getenv("HOME2") + "/datasets/ECAD/original/small_sample/";
    private static final String CALENDAR = "proleptic_gregorian";
    private static final String UNITS = "days since 1950-01-01";

    private Object readVariable(Variable variable) throws IOException {
        Class<?> dataType = variable.getDataType().getPrimitiveClassType();
        Object data;

        Attribute scaleFactorAttribute = variable.findAttribute("scale_factor");
        if (scaleFactorAttribute != null) {
            // TODO: delay the conversion to later?
            float scaleFactor = scaleFactorAttribute.getNumericValue().floatValue();
            float[] dataRaw = (float[]) variable.read().get1DJavaArray(float.class);
            for (int i = 0; i < dataRaw.length; i++) {
                dataRaw[i] *= scaleFactor;
            }
            data = dataRaw;
        } else {
            data = variable.read().get1DJavaArray(dataType);
            // time is handled manually later, so keep as int[]
        }

        return data;
    }

    private DataFrame readData(String path) throws IOException {
        NetcdfFile file = NetcdfFile.open(path);

        DataFrame df = new DataFrame();
        for (Variable variable : file.getVariables()) {
            df.add(variable.getShortName(), readVariable(variable));
        }

        return df;
    }

    private void print(DataFrame df) {
        DataFrame printData = new DataFrame(df.getSize());

        for (String column : df.keys()) {
            Object data = df.get(column);

            if (data.getClass().equals(float[].class)) {
                float[] dataRaw = (float[]) data;
                printData.put(column, Joiner.on(", ").join(Floats.asList(dataRaw)));
            } else if (data.getClass().equals(int[].class)) {
                int[] dataRaw = (int[]) data;
                String[] dates = new String[df.getSize()];
                for (int i = 0; i < df.getSize(); i++) {
                    dates[i] = DataFrame.intTimeToString(dataRaw[i], Pipeline.CALENDAR, Pipeline.UNITS);
                }
                printData.put(column, Joiner.on(", ").join(dates));
            }
        }

        Joiner.MapJoiner mapJoiner = Joiner.on("\n").withKeyValueSeparator("=");

        System.out.println(mapJoiner.join(printData.values()));
        System.out.println("");
    }

    private void head(DataFrame df, int numberRows) {
        DataFrame head = df.subset(0, numberRows);

        print(head);
    }

    private void printAggregations(DataFrame df) {
        Joiner.MapJoiner valueJoiner = Joiner.on(", ").withKeyValueSeparator("=");

        for (Map.Entry<String, Object> entry : df.values().entrySet()) {
            df.put(entry.getKey(), valueJoiner.join((Map<String, Float>) entry.getValue()));
        }

        Joiner.MapJoiner mapJoiner = Joiner.on("\n").withKeyValueSeparator("=");

        System.out.println(mapJoiner.join(df.values()));
        System.out.println("");
    }

    private Object computeAbsMaxmin(Object max, Object min) {
        float[] maxData = (float[]) max;
        float[] minData = (float[]) min;

        float[] result = new float[maxData.length];

        for (int i = 0; i < maxData.length; i++) {
            result[i] = Math.abs(maxData[i] - minData[i]);
        }

        return result;
    }

    private Object computeYearMonth(Object time) {
        int[] timeRaw = (int[]) time;
        String[] yearMonth = new String[timeRaw.length];
        for (int i = 0; i < timeRaw.length; i++) {
            yearMonth[i] = DataFrame.intTimeToString(timeRaw[i], CALENDAR, UNITS)
                    .substring(0, 7)
                    .replaceFirst("-", "");
        }

        return yearMonth;
    }

    public void start() throws IOException {
        DataFrame df1 = readData(PATH + "data1.nc");
        DataFrame df2 = readData(PATH + "data2.nc");

        // PIPELINE
        // 1. join the 2 dataframes
        DataFrame df = df1.join(df2);

        // 2. quick preview on the data
        head(df, 10);

        // 3. subset the data
        df = df.subset(709920, 1482480);

        // 4. drop rows with null values
        df = df.filter();

        // 5. drop columns
        df.pop("pp_err");
        df.pop("rr_err");

        // 6. UDF 1: compute absolute difference between max and min
        df.put("abs_diff", computeAbsMaxmin(df.get("tx"), df.get("tn")));

        // 7. explore the data through aggregations
        printAggregations(df.aggregations());

        // 8. compute mean per month
        // UDF 2: compute custom year+month format
        df.put("year_month", computeYearMonth(df.get("time")));
        // also handles the join back
        df = df.groupBy();
        df.pop("year_month");

        // careful with this! prints all
        head(df, df.getSize());
    }
}
