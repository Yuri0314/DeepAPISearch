package com.nju.util;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ParserUtil {
    public static Map<String, String> charEntity;

    static {
        charEntity = new HashMap<>();
        charEntity.put("&nbsp;", " ");
        charEntity.put("&lt;", "<");
        charEntity.put("&gt;", ">");
        charEntity.put("&amp;", "&");
        charEntity.put("&quot;", "\"");
        charEntity.put("&apos;", "'");
        charEntity.put("&copy;", "©");
        charEntity.put("&reg;", "®");
        charEntity.put("&trade;", "™");
        charEntity.put("&times;", "×");
        charEntity.put("&divide;", "÷");
    }

    /**
     * 生成新形式的方法签名字符串，形式为MethodName(paramTypes)->receiveType
     * @param methodSignature
     * @return
     */
    public static String generateMethodSignature(String methodSignature, String methodName) {
        String newSignature = methodSignature.replaceAll("<(.)*?>", "");
        int pos = newSignature.indexOf(methodName);
        return newSignature.substring(pos).trim();
    }

    /**
     * 从获取的javadoc描述字符串中获取第一句作为整体简短描述
     * @param description
     * @return
     */
    public static String generateBriefDescription(String description) {
        String strDocDescription = description.replaceAll("(\\r\\n)", " ");
        int endPos = strDocDescription.indexOf(". ");
        return endPos == -1 ? strDocDescription : strDocDescription.substring(0, endPos);
    }

    /**
     * 根据类名和方法名为没有文档描述的方法生成方法描述
     * @param className
     * @param methodName
     * @return
     */
    public static String generateDescriptionFromMethod(String className, String methodName) {
        StringBuffer sBuf = new StringBuffer();
        String regex = "([A-Z])+";
        sBuf.append(className.replaceAll(regex, " $1").toLowerCase() + " ");
        sBuf.append(methodName.replaceAll(regex, " $1").toLowerCase());
        return sBuf.toString();
    }

    /**
     * 去除带有<>的标签
     * @param description
     * @return
     */
    public static String removeTag(String description) {
        description = description.replaceAll(
                "(<code>|</code>|<a[^>]*>|</a>|<i>|</i>|</sup>|<em>|</em>|<cite>|</cite>|<p>|</p>|<b>|</b>|<I>|</I>|<pre>|</pre>|<strong>|</strong>|<blockquote>|</blockquote>)",
                "");
        description = description.replace("<sup>", "^");
        return description;
    }

    /**
     * 去除带有{\@}形式的标签
     * @param description
     * @return
     */
    public static String removeAtTag(String description) {
        String pattern = "\\{@(.)+?(\\s)+((.)*?)}";
        Matcher matcher = Pattern.compile(pattern).matcher(description);
        StringBuffer sBuffer = new StringBuffer();
        while (matcher.find()) {
            matcher.appendReplacement(sBuffer, "$3");
        }
        matcher.appendTail(sBuffer);
        return sBuffer.toString().replaceAll("\\{@(.)+?}", "");
    }

    /**
     * 替换说明中出现的所有html字符实体为对应字符
     * @param description
     * @return
     */
    public static String replaceHtmlChar(String description) {
        for (Map.Entry<String, String> entry : charEntity.entrySet())
            description = description.replaceAll(entry.getKey(), entry.getValue());
        return description;
    }
}
