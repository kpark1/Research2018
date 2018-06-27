#---------------------------------------------------------------------------------------------------
# C L A S S E S
#---------------------------------------------------------------------------------------------------
class ScopeTrace:
    """
    Class to mange a full scope trace.
    """
    undefined_value = -9999999

    def __init(self, data, n=4):
        self.data = data
        self.xvalues = []
        self.yvalues = []
        i = 0
        for line in data.split('\n'):
            f = line.split(',')
            self.xvalues.append(float(i))
            self.yvalues.append(float(f[n])
            i +=1
            
    def find_value(self,name,data,type="f"):
        value = self.undefined_value
        for line in data.split("\n"):
            f = line.split(',')
            if f[0] == name:
                if   type == 'f':
                    value = float(f[1])
                    #print " Value[%s]  %f (F)"%(name,value)
                elif type == 'i':
                    value = int(f[1])
                    #print " Value[%s]  %d (I)"%(name,value)
                else:
                    value = f[1]
                    #print " Value[%s]  %s (S)"%(name,value)
                break
        return value

    def find_baseline_and_jitter(self,xmin,xmax):
        n = 0
        sum = 0
        sum2 = 0
        for x,y in zip(self.xvalues,self.yvalues):
            if x>=xmin and x<xmax:
                sum = sum + y
                sum2 = sum2 + y*y
                n = n + 1

        baseline = 0
        jitter = 0
        if n>0:
            baseline = sum/n
            jitter = sum2/n - baseline*baseline

        return (baseline,jitter)

    def find_number_of_pulses(self,baseline,threshold,delta_min):
        n_pulses_found = 0
        last_y = self.yvalues[0]
        latched = False
        for y in self.yvalues:
            delta_y = last_y - y
            if   y<baseline-threshold:
                #print " D: %f,  Dmin: %f"%(delta_y,delta_min)
                if delta_y>delta_min and not latched:
                    n_pulses_found += 1
                    latched = True
            elif y>baseline-threshold-delta_min:
                latched = False
            last_y = y

                
        return n_pulses_found 
