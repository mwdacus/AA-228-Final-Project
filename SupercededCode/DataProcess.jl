#Michael Dacus                                    AA 228-Final Project                        Team 34
# Ian Hokai
# Tyler Weiss
#
#Objective: Reconfigure the raw dataset into State(s),Action(s),reward(r),next state(sp)
#Input: Incoming Raw Data (D)
#Output: (s,a,r,sp) (for each quarter)
#----------------------------------------------------------------------------------
using DataFrames
using DataFramesMeta
using CSV
#Import CSV File
D=DataFrame(CSVFile.File("rawdata.csv"))
#Sort for Extra Points and 2 Point Conversions
D_tn=@where(D,in(["extra_point","2_point"]).(:type))
#Sort only by home team touchdowns

#Sort by Quarter
D1=@where(D_tn,in([1]).(:quarter))
D2=@where(D_tn,in([2]).(:quarter))
D3=@where(D_tn,in([3]).(:quarter))
D4=@where(D_tn,in([4]).(:quarter))

#Initialize Matrix for Each Quarter
sarsp_q1=DataFrame(s=Int64[],a=Int64[],r=Int64[],sp=Int64[]);
sarsp_q2=DataFrame(s=Int64[],a=Int64[],r=Int64[],sp=Int64[]);
sarsp_q3=DataFrame(s=Int64[],a=Int64[],r=Int64[],sp=Int64[]);
sarsp_q4=DataFrame(s=Int64[],a=Int64[],r=Int64[],sp=Int64[]);

function Discretize(D,sarsp)
    (row,col)=size(D);
    for i in 1:row
        sp=D.home_points[i]-D.away_points[i]
        if occursin("extra point",D.description[i])==true
            a=1;
            r=1;
        else
            a=2;
            r=2;
        end
        if a==1 && occursin("good",D.description[i])==true
            s=(D.home_points[i]-1)-D.away_points #Calculate previous state (before getting extra point)
        elseif a==2 && occursin("good",D.description[i])==true
            s=(D.home_points[i]-2)-D.away_points #Calculate the previous state (before getting two point conversion)
        else
            s=sp;
        end
        push!(sarsp,[s,a,r,sp])
    end
    return sarsp
end

#Output the sarsp's for each quarter:
sarsp_q1=Discretize(D1,sarsp_q1)
sarsp_q2=Discretize(D2,sarsp_q2)
sarsp_q3=Discretize(D3,sarsp_q3)
sarsp_q4=Discretize(D4,sarsp_q4)
